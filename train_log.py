import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


LINE_PATTERN = re.compile(
    r"^\s*(?P<time>\d{4}-\d{2}-\d{2} [0-9:.]+) \| "
    r"epoch=(?P<epoch>\d+) \| "
    r"train_loss=(?P<train>[0-9.]+) \| "
    r"valid_(?:loss|dice)=(?P<valid>[0-9.]+) \| "
    r"is_best=(?P<best>True|False)\s*$"
)


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values[:]

    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        if i >= window - 1:
            out.append(acc / window)
        else:
            out.append(v)
    return out


def parse_log(log_path):
    epochs = []
    train_losses = []
    valid_losses = []
    is_best_flags = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = LINE_PATTERN.match(line)
            if not m:
                continue
            epochs.append(int(m.group("epoch")))
            train_losses.append(float(m.group("train")))
            valid_losses.append(float(m.group("valid")))
            is_best_flags.append(m.group("best") == "True")

    if not epochs:
        raise RuntimeError(f"No epoch records parsed from log: {log_path}")

    return epochs, train_losses, valid_losses, is_best_flags


def summarize(epochs, train_losses, valid_losses, is_best_flags):
    best_idx = max(range(len(valid_losses)), key=lambda i: valid_losses[i])
    first_best_idx = next((i for i, flag in enumerate(is_best_flags) if flag), None)

    result = {
        "num_epochs": len(epochs),
        "epoch_start": epochs[0],
        "epoch_end": epochs[-1],
        "best_valid_dice": valid_losses[best_idx],
        "best_valid_epoch": epochs[best_idx],
        "train_loss_at_best": train_losses[best_idx],
        "first_flagged_best_epoch": epochs[first_best_idx] if first_best_idx is not None else None,
        "last_train_loss": train_losses[-1],
        "last_valid_dice": valid_losses[-1],
    }
    return result


def plot_curves(
    epochs,
    train_losses,
    valid_losses,
    is_best_flags,
    output_png,
    smooth_window=1,
):
    train_plot = moving_average(train_losses, smooth_window)
    valid_plot = moving_average(valid_losses, smooth_window)

    plt.figure(figsize=(11, 6))
    plt.plot(epochs, train_plot, label="train_loss", linewidth=2)
    plt.plot(epochs, valid_plot, label="valid_loss", linewidth=2)

    best_points_x = [e for e, f in zip(epochs, is_best_flags) if f]
    best_points_y = [v for v, f in zip(valid_losses, is_best_flags) if f]
    if best_points_x:
        plt.scatter(best_points_x, best_points_y, label="is_best=True", s=28, marker="o")

    best_idx = max(range(len(valid_losses)), key=lambda i: valid_losses[i])
    plt.scatter(
        [epochs[best_idx]],
        [valid_losses[best_idx]],
        label=f"global best: epoch {epochs[best_idx]}",
        s=80,
        marker="*",
        zorder=5,
    )

    plt.title("Training Log Visualization")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize PraNet train.log")
    parser.add_argument(
        "log_path_pos",
        nargs="?",
        default="",
        help="Optional positional path to train.log (quick usage)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="",
        help="Path to train.log",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="",
        help="Output figure path",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving average window for plotting (>=1)",
    )
    args = parser.parse_args()

    default_log = "/SSD2/pengzhipeng/segcode/PraNet-master/checkpoint/run2_200epoch/train.log"
    log_input = args.log_path_pos or args.log_path or default_log
    log_path = Path(log_input).resolve()

    if args.out_png:
        out_png = Path(args.out_png).resolve()
    else:
        out_png = log_path.with_name("train_log_curve.png")

    if args.out_json:
        out_json = Path(args.out_json).resolve()
    else:
        out_json = log_path.with_name("train_log_summary.json")

    epochs, train_losses, valid_losses, is_best_flags = parse_log(log_path)
    summary = summarize(epochs, train_losses, valid_losses, is_best_flags)

    plot_curves(
        epochs,
        train_losses,
        valid_losses,
        is_best_flags,
        out_png,
        smooth_window=max(1, args.smooth_window),
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] figure:", out_png)
    print("[OK] summary:", out_json)
    print("[INFO]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
