import torch
from torch.autograd import Variable
import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime
from lib.RePraNet import RePraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, AvgMeter
import torch.nn.functional as F


def structure_loss(pred, mask):
    # 1. 正常的 BCE (求平均，数值稳定)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    
    # 2. 正常的 IoU
    pred_soft = torch.sigmoid(pred)
    inter = (pred_soft * mask).sum(dim=(2, 3))
    union = (pred_soft + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    iou = iou.mean()

    edge_zone = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    edge_error = torch.abs(pred_soft - mask) 
    loss_edge = (edge_error * edge_zone).sum(dim=(2, 3)) / (edge_zone.sum(dim=(2, 3)) + 1e-6)
    loss_edge = loss_edge.mean()
    
    return bce + iou + 0.5 * loss_edge


def _to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _dice_score(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice score for a batch."""
    pred_bin = (pred > threshold).float()
    gt_bin = gt.float()
    inter = (pred_bin * gt_bin).flatten(1).sum(dim=1)
    union = pred_bin.flatten(1).sum(dim=1) + gt_bin.flatten(1).sum(dim=1)
    dice = 2.0 * inter / (union + 1e-6)
    return dice.mean().item()


def validate_one_epoch(val_loader, model):
    """Run one validation epoch and return mean Dice score (higher is better)."""
    model.eval()
    dice_sum = 0.0
    sample_count = 0

    with torch.inference_mode():
        for images, gts in val_loader:
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(images)
            res = torch.sigmoid(lateral_map_1)

            bs = images.size(0)
            dice_sum += _dice_score(res, gts) * bs
            sample_count += bs

    model.train()
    if sample_count == 0:
        return 0.0
    return dice_sum / sample_count


class CheckpointManager:
    """Create run directories and manage best/last checkpoint saving."""

    def __init__(self, total_epochs, config=None, snapshot_start_epoch=10, patience=10):
        project_root = Path(__file__).resolve().parent
        self.checkpoint_root = str(project_root / "checkpoint")
        self.total_epochs = total_epochs
        self.config = config or {}
        self.snapshot_start_epoch = snapshot_start_epoch
        self.patience = max(int(patience), 1)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.run_id = self._next_run_id()
        self.run_name = f"run{self.run_id}_{self.total_epochs}epoch"
        self.run_dir = os.path.join(self.checkpoint_root, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.best_model_path = os.path.join(self.run_dir, "best_model.pth")
        self.last_model_path = os.path.join(self.run_dir, "last_model.pth")
        self.log_path = os.path.join(self.run_dir, "train.log")
        self.summary_path = os.path.join(self.run_dir, "checkpoint_summary.json")

        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.snapshot_counter = 0
        self.epochs_without_improvement = 0

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("#" * 80 + "\n")
            f.write(f"start_time: {datetime.now()}\n")
            f.write(f"run_dir: {self.run_dir}\n")
            f.write(f"config: {json.dumps(self.config, ensure_ascii=True)}\n")

    def _next_run_id(self):
        max_id = 0
        pattern = re.compile(r"^run(\d+)(?:[_-]\d+epoch[s]?)?$")
        for name in os.listdir(self.checkpoint_root):
            full_path = os.path.join(self.checkpoint_root, name)
            if not os.path.isdir(full_path):
                continue
            m = pattern.match(name)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def save_last(self, model):
        torch.save(model.state_dict(), self.last_model_path)

    def record_validation(self, metric):
        """Record one validation metric and return whether it improved."""
        if metric > self.best_metric:
            self.best_metric = metric
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    def should_stop(self):
        return self.epochs_without_improvement >= self.patience

    def maybe_save_best(self, model, metric, epoch):
        if self.record_validation(metric):
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.best_model_path)
            if epoch >= self.snapshot_start_epoch:
                self.snapshot_counter += 1
                snapshot_path = os.path.join(self.run_dir, f"best_model{self.snapshot_counter}.pth")
                torch.save(model.state_dict(), snapshot_path)
            return True
        return False

    def log_epoch(self, epoch, train_loss, valid_dice=None, is_best=False):
        msg = (
            f"{datetime.now()} | epoch={epoch} | "
            f"train_loss={train_loss:.6f}"
        )
        if valid_dice is not None:
            msg += f" | valid_dice={valid_dice:.6f}"
        msg += f" | is_best={is_best}\n"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg)

    def finalize(self):
        if self.best_epoch < 0 and os.path.exists(self.last_model_path):
            # If validation was disabled, keep a usable best checkpoint alias.
            torch.save(torch.load(self.last_model_path), self.best_model_path)

        payload = {
            "run_dir": self.run_dir,
            "best_model": self.best_model_path,
            "last_model": self.last_model_path,
            "best_metric": None if self.best_epoch < 0 else self.best_metric,
            "best_epoch": None if self.best_epoch < 0 else self.best_epoch,
            "end_time": str(datetime.now()),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            current_batch_size = images.size(0)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts) 
            loss1 = structure_loss(lateral_map_1, gts)
            loss = 0.2*loss5 + 0.1*loss4 + 0.2*loss3 + 0.3*loss2 + 0.4*loss1     # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, 0.5)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.detach(), current_batch_size)
                loss_record2.update(loss2.detach(), current_batch_size)
                loss_record3.update(loss3.detach(), current_batch_size)
                loss_record4.update(loss4.detach(), current_batch_size)
                loss_record5.update(loss5.detach(), current_batch_size)
        # ---- train visualization ----
        if i == 1 or i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:0.4f}, lateral-2: {:0.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    epoch_train_loss = (
        loss_record1.show() + loss_record2.show() + loss_record3.show() + loss_record4.show() + loss_record5.show()
    ) / 5.0
    return float(epoch_train_loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=25, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--lora-rank', type=int, choices=(8, 16, 32, 64, 128),
                        default=8, help='LoRA rank for SAM2 Hiera attention')
    parser.add_argument('--lora-alpha', type=float, default=None,
                        help='LoRA scaling alpha; defaults to 2 * lora-rank')
    parser.add_argument('--lora-dropout', type=float, default=0.0,
                        help='LoRA dropout, default 0 for deterministic comparison')
    parser.add_argument('--train-path', type=str,
                        default='/HDD/pengzhipeng/dataset/TrainDataset', help='path to train dataset')
    parser.add_argument('--split-dir', type=str,
                        default='utils/TrainDataset', help='split directory with train.txt/val.txt')
    parser.set_defaults(use_augmentation=True)
    parser.add_argument('--use-augmentation', dest='use_augmentation', action='store_true',
                        help='enable albumentations-based training augmentation (default: enabled)')
    parser.add_argument('--no-augmentation', dest='use_augmentation', action='store_false',
                        help='disable albumentations-based training augmentation')
    parser.add_argument('--valid-interval', type=int,
                        default=1, help='run validation every N epochs, default 1')
    parser.add_argument('--patience', type=int,
                        default=10, help='early stopping patience in validation checks, default 10')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = RePraNet(
        lora_rank=opt.lora_rank,
        lora_alpha=opt.lora_alpha,
        lora_dropout=opt.lora_dropout,
    ).cuda()
    total_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        '[LoRA] rank={} alpha={} targets={} lora_parameters={} total_trainable={}'.format(
            model.lora_rank,
            model.lora_alpha,
            len(model.lora_target_names),
            model.lora_parameter_count,
            total_trainable,
        )
    )

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = filter(lambda parameter: parameter.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(opt.epoch - 1, 1),
    )

    split_dir = Path(opt.split_dir)
    if not split_dir.is_absolute():
        script_dir = Path(__file__).resolve().parent
        split_dir = (script_dir / split_dir).resolve()

    train_split = split_dir / 'train.txt'
    val_split = split_dir / 'val.txt'

    print('[Data] split mode enabled.')
    print('[Data] dataset root:', opt.train_path)
    print('[Data] train split:', train_split)

    train_loader = get_loader(
        image_root=None,
        gt_root=None,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        split_file=str(train_split),
        dataset_root=opt.train_path,
        use_augmentation=opt.use_augmentation,
    )

    if val_split.exists():
        val_loader = get_loader(
            image_root=None,
            gt_root=None,
            batchsize=opt.batchsize,
            trainsize=opt.trainsize,
            shuffle=False,
            split_file=str(val_split),
            dataset_root=opt.train_path,
            use_augmentation=False,
        )
        print('[Data] val split:', val_split)
    else:
        val_loader = None
        print('[Warning] split mode is enabled but val.txt is missing. Validation disabled.')

    total_step = len(train_loader)

    ckpt_manager = CheckpointManager(
        total_epochs=opt.epoch,
        config=vars(opt),
        patience=opt.patience,
    )
    print('[Checkpoint Run Dir]:', ckpt_manager.run_dir)

    print("#"*20, "Start Training", "#"*20)
    print('[LR Scheduler] cosine annealing with initial lr {:.2e}'.format(opt.lr))

    for epoch in range(1, opt.epoch + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print('[LR] Epoch {} lr={:.8f}'.format(epoch, current_lr))
        train_loss = train(train_loader, model, optimizer, epoch)

        valid_dice = None
        is_best = False
        if val_loader is not None and epoch % max(opt.valid_interval, 1) == 0:
            valid_dice = validate_one_epoch(val_loader, model)
            is_best = ckpt_manager.maybe_save_best(model, valid_dice, epoch)
            print('[Validation] Epoch {} valid_dice={:.6f} is_best={} no_improvement={}/{}'.format(
                epoch,
                valid_dice,
                is_best,
                ckpt_manager.epochs_without_improvement,
                ckpt_manager.patience,
            ))

        scheduler.step()

        ckpt_manager.log_epoch(epoch, train_loss, valid_dice, is_best)

        if valid_dice is not None and ckpt_manager.should_stop():
            print('[Early Stopping] Epoch {}: validation did not improve for {} checks.'.format(
                epoch,
                ckpt_manager.patience,
            ))
            break

    ckpt_manager.save_last(model)
    ckpt_manager.finalize()
    print('[Training Complete] best_model:', ckpt_manager.best_model_path)
    print('[Training Complete] last_model:', ckpt_manager.last_model_path)
