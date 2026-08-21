import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def infer_dataset_name(dataset_root: Path) -> str:
    name = dataset_root.resolve().name.strip()
    if not name:
        name = "dataset"
    return name


def is_dataset_root(root: Path) -> bool:
    return (root / "images").is_dir() and (root / "masks").is_dir()


def discover_dataset_roots(root: Path) -> List[Tuple[str, Path]]:
    """Resolve one root to one or many dataset roots.

    Returns tuples of (output_relative_name, dataset_root_path).
    """
    root = root.resolve()

    if is_dataset_root(root):
        return [(infer_dataset_name(root), root)]

    children = [p for p in sorted(root.iterdir()) if p.is_dir() and is_dataset_root(p)]
    if children:
        parent_name = infer_dataset_name(root)
        return [(f"{parent_name}/{infer_dataset_name(child)}", child.resolve()) for child in children]

    raise FileNotFoundError(
        f"No dataset found under {root}. Expected either images/masks in root or subfolders each containing images/masks."
    )


def collect_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_map: Dict[str, Path] = {
        p.stem: p
        for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }
    mask_map: Dict[str, Path] = {
        p.stem: p
        for p in sorted(mask_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }

    common = sorted(set(image_map.keys()) & set(mask_map.keys()))
    if not common:
        raise RuntimeError("No matched image/mask pairs found by filename stem.")

    pairs = [(image_map[name], mask_map[name]) for name in common]
    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}.")

    total = len(pairs)
    required_subsets = sum([train_ratio > 0, val_ratio > 0, test_ratio > 0])
    if total < required_subsets:
        raise RuntimeError(
            f"Need at least {required_subsets} matched pairs for configured split, got {total}."
        )

    random.Random(seed).shuffle(pairs)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = pairs[:train_end]
    val_set = pairs[train_end:val_end]
    test_set = pairs[val_end:]

    if train_ratio > 0 and len(train_set) == 0:
        raise RuntimeError("Split produced empty train subset.")
    if val_ratio > 0 and len(val_set) == 0:
        raise RuntimeError("Split produced empty val subset.")
    if test_ratio > 0 and len(test_set) == 0:
        raise RuntimeError("Split produced empty test subset.")

    return train_set, val_set, test_set


def resolve_split_policy(root: Path) -> Tuple[float, float, float]:
    """Return (train_ratio, val_ratio, test_ratio) for a given root input."""
    if root.resolve().name == "TrainDataset":
        return 1295 / 1450, 155 / 1450, 0.0
    return 0.8, 0.1, 0.1


def write_split_file(output_file: Path, subset: List[Tuple[Path, Path]], dataset_root: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{img.relative_to(dataset_root)} {mask.relative_to(dataset_root)}" for img, mask in subset]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split one or multiple datasets into train/val/test by 8:1:1.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/HDD/pengzhipeng/dataset/Kvasir-SEG"),
        help="Single dataset root that contains images/ and masks/.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        type=Path,
        default=None,
        help="Multiple dataset roots. If set, this overrides --dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory for split cache folders. Default is current utils/ directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files if they already exist.",
    )
    args = parser.parse_args()

    if args.dataset_roots:
        dataset_roots = args.dataset_roots
    else:
        dataset_roots = [args.dataset_root]

    base_output_dir = args.output_dir.resolve()

    for root in dataset_roots:
        train_ratio, val_ratio, test_ratio = resolve_split_policy(root)
        discovered = discover_dataset_roots(root)

        for dataset_name, dataset_root in discovered:
            output_dir = base_output_dir / dataset_name
            train_out = output_dir / "train.txt"
            val_out = output_dir / "val.txt"
            test_out = output_dir / "test.txt"

            required_files = [train_out, val_out] if test_ratio == 0 else [train_out, val_out, test_out]
            if (not args.overwrite) and all(p.exists() for p in required_files):
                print("=" * 60)
                print(f"Dataset root: {dataset_root}")
                print(f"Dataset name: {dataset_name}")
                print(f"Reuse existing split files: {output_dir}")
                continue

            pairs = collect_pairs(dataset_root)
            train_set, val_set, test_set = split_pairs(
                pairs,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=args.seed,
            )

            write_split_file(train_out, train_set, dataset_root)
            write_split_file(val_out, val_set, dataset_root)
            if test_ratio > 0:
                write_split_file(test_out, test_set, dataset_root)
            elif args.overwrite and test_out.exists():
                test_out.unlink()

            print("=" * 60)
            print(f"Dataset root: {dataset_root}")
            print(f"Dataset name: {dataset_name}")
            print(f"Total matched pairs: {len(pairs)}")
            if test_ratio > 0:
                print(f"Split ratio: {train_ratio}:{val_ratio}:{test_ratio}")
                print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
            else:
                print(f"Split ratio: {train_ratio}:{val_ratio}")
                print(f"Train: {len(train_set)} | Val: {len(val_set)}")
            print(f"Split files saved to: {output_dir}")


if __name__ == "__main__":
    main()