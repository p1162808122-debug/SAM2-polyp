"""Run a short, comparable update sequence for multiple SAM2 LoRA ranks."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch

from lib.RePraNet import RePraNet
from MyTrain import structure_loss
from utils.dataloader import get_loader


SUPPORTED_RANKS = (8, 16, 32, 64, 128)


def build_rank_plan(ranks: Sequence[int]) -> Tuple[Tuple[int, float], ...]:
    ranks = tuple(int(rank) for rank in ranks)
    if not ranks:
        raise ValueError("at least one LoRA rank is required")
    invalid = tuple(rank for rank in ranks if rank not in SUPPORTED_RANKS)
    if invalid:
        raise ValueError(f"unsupported LoRA ranks: {invalid}; choose from {SUPPORTED_RANKS}")
    return tuple((rank, float(2 * rank)) for rank in ranks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repranet_loss(outputs, gts):
    loss5, loss4, loss3, loss2, loss1 = (
        structure_loss(output, gts) for output in outputs
    )
    return 0.2 * loss5 + 0.1 * loss4 + 0.2 * loss3 + 0.3 * loss2 + 0.4 * loss1


def lora_gradient_norm(model: RePraNet) -> float:
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if ".lora_" in name and parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().pow(2).sum().item())
    return squared_norm**0.5


def cuda_memory_report(device: torch.device):
    if device.type != "cuda":
        return {}
    return {
        "peak_memory_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2
        ),
        "peak_memory_reserved_mb": round(
            torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2
        ),
    }


def train_rank(
    rank: int,
    alpha: float,
    loader,
    steps: int,
    device: torch.device,
    lr: float,
    seed: int,
):
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    model = RePraNet(lora_rank=rank, lora_alpha=alpha).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )
    total_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    losses = []
    last_grad_norm = 0.0
    iterator = iter(loader)
    try:
        for _ in range(steps):
            try:
                images, gts = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, gts = next(iterator)
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = repranet_loss(outputs, gts)
            loss.backward()
            last_grad_norm = lora_gradient_norm(model)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        return {
            "rank": rank,
            "alpha": alpha,
            "steps": steps,
            "status": "ok",
            "lora_targets": len(model.lora_target_names),
            "lora_parameters": model.lora_parameter_count,
            "total_trainable_parameters": total_trainable,
            "loss_history": losses,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "last_lora_gradient_norm": last_grad_norm,
            "elapsed_seconds": time.perf_counter() - start_time,
            **cuda_memory_report(device),
        }
    except Exception as error:
        return {
            "rank": rank,
            "alpha": alpha,
            "steps": steps,
            "status": "error",
            "error": repr(error),
            "elapsed_seconds": time.perf_counter() - start_time,
            **cuda_memory_report(device),
        }
    finally:
        del optimizer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(SUPPORTED_RANKS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--train-path",
        default="/HDD/pengzhipeng/dataset/TrainDataset",
    )
    parser.add_argument(
        "--split-file",
        default=str(project_root / "utils/TrainDataset/train.txt"),
    )
    parser.add_argument("--output", default="lora_rank_validation.json")
    return parser.parse_args()


def write_report(output_path: Path, report) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    temporary_path.replace(output_path)


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    loader = get_loader(
        image_root=None,
        gt_root=None,
        batchsize=args.batchsize,
        trainsize=args.trainsize,
        shuffle=False,
        split_file=args.split_file,
        dataset_root=args.train_path,
        use_augmentation=False,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(args),
        "results": [],
    }
    results = []
    for rank, alpha in build_rank_plan(args.ranks):
        print(f"[Rank sweep] rank={rank} alpha={alpha}")
        result = train_rank(
            rank=rank,
            alpha=alpha,
            loader=loader,
            steps=args.steps,
            device=device,
            lr=args.lr,
            seed=args.seed,
        )
        results.append(result)
        report["results"] = results
        write_report(output_path, report)
        print(json.dumps(result, ensure_ascii=False))

    print(f"[Rank sweep] report={output_path.resolve()}")


if __name__ == "__main__":
    main()
