import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import re
from pathlib import Path
import imageio
from lib.RePraNet import RePraNet
from utils.dataloader import test_dataset


def load_torch_file(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def discover_model_weights(run_dir: str):
    """Return last_model and the newest available best checkpoint."""
    run_path = Path(run_dir)
    last_model = run_path / 'last_model.pth'
    best_model = run_path / 'best_model.pth'
    numbered_best = []
    numbered_pattern = re.compile(r'^best_model(\d+)$')

    for f in run_path.iterdir():
        if f.suffix != '.pth':
            continue
        match = numbered_pattern.match(f.stem)
        if match:
            numbered_best.append((int(match.group(1)), f))

    weights = []
    if last_model.is_file():
        weights.append(('last_model', str(last_model)))

    if numbered_best:
        _, newest_best = max(numbered_best, key=lambda item: item[0])
        weights.append((newest_best.stem, str(newest_best)))
    elif best_model.is_file():
        weights.append(('best_model', str(best_model)))

    return weights


def _make_result_subdir(run_name: str, model_stem: str) -> Path:
    """results/<run_name>/<model_stem>/"""
    project_root = Path(__file__).resolve().parent
    return project_root / 'results' / run_name / model_stem


def discover_latest_run() -> Path:
    """Find the checkpoint directory with the largest run ID number."""
    project_root = Path(__file__).resolve().parent
    checkpoint_root = project_root / "checkpoint"
    if not checkpoint_root.is_dir():
        return None

    pattern = re.compile(r"^run(\d+)(?:[_-]\d+epoch[s]?)?$")
    latest_id = -1
    latest_dir = None
    for name in os.listdir(str(checkpoint_root)):
        full_path = checkpoint_root / name
        if not full_path.is_dir():
            continue
        m = pattern.match(name)
        if m:
            run_id = int(m.group(1))
            if run_id > latest_id:
                latest_id = run_id
                latest_dir = full_path
    return latest_dir


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--lora-rank', type=int, choices=(8, 16, 32, 64, 128), default=8,
                    help='LoRA rank used to create the checkpoint model')
parser.add_argument('--lora-alpha', type=float, default=None,
                    help='LoRA alpha; defaults to 2 * lora-rank')
parser.add_argument(
    '--run-dir',
    type=str,
    default=None,
    help=(
        'Path to a checkpoint run directory (e.g. checkpoint/run5_100epoch). '
        'At most two weights are tested: last_model and the newest numbered '
        'best snapshot (falling back to best_model). '
        '(default: auto-detect the latest runN directory)'
    )
)
parser.add_argument(
    '--pth-path',
    type=str,
    default=None,
    help='Path to a single model weight file. Overrides --run-dir if provided.'
)
parser.add_argument(
    '--test-path',
    type=str,
    default='/HDD/pengzhipeng/dataset/TestDataset',
    help='test dataset root, expected structure: <root>/<dataset>/{images,masks}'
)
parser.add_argument(
    '--datasets',
    nargs='+',
    default=['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'],
    help='dataset names under test_path'
)

opt = None
test_path = None


def test_one_model(pth_path: str, pth_stem: str, run_name: str):
    """Load a model, run inference on all datasets, save results."""
    print(f"\n{'='*60}")
    print(f"[Test] model: {pth_stem}  ({pth_path})")
    print(f"{'='*60}")

    model = RePraNet(lora_rank=opt.lora_rank, lora_alpha=opt.lora_alpha)
    model.load_state_dict(load_torch_file(pth_path))
    model.cuda()
    model.eval()

    results_base = _make_result_subdir(run_name, pth_stem)

    for data_name in opt.datasets:
        data_path = test_path / data_name
        if not data_path.exists():
            print(f"  [Skip] {data_name} — not found at {data_path}")
            continue

        save_path = results_base / data_name
        os.makedirs(save_path, exist_ok=True)
        print(f"  [Dataset] {data_name}  ->  {save_path}")

        image_root = str(data_path / 'images') + '/'
        gt_root = str(data_path / 'masks') + '/'
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            with torch.inference_mode():
                res5, res4, res3, res2, res1 = model(image)
                res = res1
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = torch.sigmoid(res)
            res = res.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res_u8 = (res * 255.0).astype(np.uint8)
            imageio.imwrite(str(save_path / name), res_u8)

    print(f"  [Done] {pth_stem}")


def main():
    global opt, test_path
    opt = parser.parse_args()
    test_path = Path(opt.test_path)

    # ── Entry point ────────────────────────────────────────────────────────────
    if opt.pth_path:
        # Single-weight mode (backward compatible)
        pth_path = str(Path(opt.pth_path).resolve())
        pth_stem = Path(pth_path).stem
        run_name = pth_stem
        test_one_model(pth_path, pth_stem, run_name)

    elif opt.run_dir:
        run_dir = Path(opt.run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--run-dir not found: {run_dir}")

        # Extract run name (e.g. "run5_100epoch")
        run_name = run_dir.name

        weights = discover_model_weights(str(run_dir))
        if not weights:
            print(f"No .pth files found in {run_dir}")
        else:
            print(f"Found {len(weights)} model(s) in {run_dir}:")
            for stem, path in weights:
                print(f"  {stem}  ({path})")
            print()

            for pth_stem, pth_path in weights:
                test_one_model(pth_path, pth_stem, run_name)

    else:
        # Auto-detect latest run directory
        latest = discover_latest_run()
        if latest is None:
            print("Error: no checkpoint directory found and --run-dir not specified.")
            print("Tip: specify --run-dir explicitly, e.g. --run-dir ./checkpoint/run1_25epoch")
            raise SystemExit(1)
        print(f"[Auto] Latest run detected: {latest}")
        run_dir = latest
        run_name = run_dir.name

        weights = discover_model_weights(str(run_dir))
        if not weights:
            print(f"No .pth files found in {run_dir}")
        else:
            print(f"Found {len(weights)} model(s) in {run_dir}:")
            for stem, path in weights:
                print(f"  {stem}  ({path})")
            print()

            for pth_stem, pth_path in weights:
                test_one_model(pth_path, pth_stem, run_name)


if __name__ == '__main__':
    main()
