import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-12
THRESHOLD = 0.5


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < EPS:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def resize_like(src: np.ndarray, target_shape) -> np.ndarray:
    h, w = target_shape
    img = Image.fromarray(src.astype(np.uint8))
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.array(img)


def find_gt_path(gt_dir: Path, pred_name: str) -> Path:
    direct = gt_dir / pred_name
    if direct.exists():
        return direct

    stem = Path(pred_name).stem
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        path = gt_dir / f"{stem}{ext}"
        if path.exists():
            return path

    raise FileNotFoundError(f"GT not found for prediction: {pred_name} under {gt_dir}")


def evaluate_one_dataset(pred_dataset_dir: Path, dataset: str, data_path: Path):
    """Evaluate one dataset using a fixed threshold of 0.5 and return mean Dice."""
    gt_path = data_path / dataset / "masks"
    pred_files = sorted(pred_dataset_dir.glob("*.png"))
    if not pred_files:
        raise RuntimeError(f"No prediction png files found in {pred_dataset_dir}")

    dice_scores = []

    for pred_path in pred_files:
        gt_mask = find_gt_path(gt_path, pred_path.name)
        gt = read_mask(gt_mask) > 128

        pred_img = read_mask(pred_path)
        if pred_img.shape != gt.shape:
            pred_img = resize_like(pred_img, gt.shape)

        resmap = normalize_map(pred_img.astype(np.float64) / 255.0)
        pred_binary = resmap >= THRESHOLD

        gt_pixels = np.count_nonzero(gt)
        pred_pixels = np.count_nonzero(pred_binary)
        intersection = np.count_nonzero(pred_binary & gt)
        dice = 2.0 * intersection / (gt_pixels + pred_pixels + EPS)
        dice_scores.append(dice)

    return {"meanDic": float(np.mean(dice_scores))}


def find_pred_dataset_dir(
    results_root: Path,
    run_name: str,
    model_subfolder: str,
    dataset: str,
) -> Path:
    """
    Search order:
      results/run/model_subfolder/dataset
      results/run/model_subfolder/result1/dataset
      results/run/model_subfolder/test1/dataset
    Falls back to the old flat layout.
    """
    run_dir = results_root / run_name
    candidates = [
        run_dir / model_subfolder / dataset,
        run_dir / model_subfolder / "result1" / dataset,
        run_dir / model_subfolder / "test1" / dataset,
        results_root / model_subfolder / dataset,
        results_root / model_subfolder / "result1" / dataset,
        results_root / model_subfolder / "test1" / dataset,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return candidates[0]


def auto_discover_latest_run(results_root: Path):
    """Return the latest run folder and its best/last model subfolders."""
    run_pattern = re.compile(r"^run(\d+)_", re.IGNORECASE)
    candidates = []

    if not results_root.exists():
        return None, None

    for path in results_root.iterdir():
        if path.is_dir():
            match = run_pattern.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path.name))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    run_name = candidates[0][1]
    run_dir = results_root / run_name

    model_pattern = re.compile(r"^(?:last_model|best_model\d*)$", re.IGNORECASE)
    model_subfolders = [
        path.name
        for path in sorted(run_dir.iterdir())
        if path.is_dir() and model_pattern.match(path.name)
    ]

    if not model_subfolders:
        model_subfolders = [run_name]

    return run_name, model_subfolders


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Evaluate segmentation predictions at fixed threshold 0.5 using mean Dice."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/HDD/pengzhipeng/dataset/TestDataset",
        help="GT root path: <data-path>/<dataset>/masks/",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Run folder name under <project>/results (e.g. 'run1_20epoch'). "
            "If omitted, the run with the largest runN is selected automatically."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "CVC-300",
            "CVC-ClinicDB",
            "CVC-ColonDB",
            "ETIS-LaribPolypDB",
            "Kvasir",
        ],
        help="Datasets to evaluate",
    )
    args = parser.parse_args()

    results_root = script_dir / "results"
    if not results_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {results_root}")

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (script_dir / data_path).resolve()

    if args.models is None:
        run_name, model_subfolders = auto_discover_latest_run(results_root)
        if run_name is None:
            raise RuntimeError(
                f"No runN_xxepoch folder found in {results_root}. "
                "Please specify --models manually."
            )
        print(f"[Auto] Discovered run: {run_name}")
        print(f"[Auto] Auto-discovered model subfolders: {model_subfolders}")
    else:
        run_name = args.models
        run_dir = results_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")

        model_pattern = re.compile(r"^(?:last_model|best_model\d*)$", re.IGNORECASE)
        model_subfolders = [
            path.name
            for path in sorted(run_dir.iterdir())
            if path.is_dir() and model_pattern.match(path.name)
        ]
        if not model_subfolders:
            model_subfolders = [run_name]

        print(f"[Run] Using run: {run_name}")
        print(f"[Run] Model subfolders: {model_subfolders}")

    for model_subfolder in model_subfolders:
        out_root = script_dir / "EvaluateResults" / run_name / model_subfolder
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Run={run_name}  Model={model_subfolder}")
        print(f"Threshold:       {THRESHOLD}")
        print(f"Prediction root: {results_root}")
        print(f"Output root:     {out_root}")
        print(f"{'=' * 60}")

        result_blocks = []
        for dataset in args.datasets:
            pred_dataset_dir = find_pred_dataset_dir(
                results_root, run_name, model_subfolder, dataset
            )

            if not pred_dataset_dir.exists():
                print(f"  [Skip] dataset={dataset}: missing {pred_dataset_dir}")
                continue

            print(f"  Evaluating dataset={dataset} ...")
            result = evaluate_one_dataset(pred_dataset_dir, dataset, data_path)

            block_lines = [
                f"Run       : {run_name}",
                f"Model     : {model_subfolder}",
                f"Dataset   : {dataset}",
                f"Threshold : {THRESHOLD}",
                f"meanDic   : {result['meanDic']:.3f}",
            ]
            block = "\n".join(block_lines)
            print(block)
            result_blocks.append(block + "\n" + "-" * 48 + "\n")

        if result_blocks:
            txt_path = out_root / f"{model_subfolder}_result.txt"
            txt_path.write_text("".join(result_blocks), encoding="utf-8")


if __name__ == "__main__":
    main()
