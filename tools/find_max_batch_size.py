"""Find the largest stable even batch size for RePraNet3 in isolated workers."""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.RePraNet import RePraNet
from MyTrain import structure_loss
from utils.dataloader import get_loader
from utils.utils import clip_gradient


SIZE_RATES = (0.75, 1.0, 1.25)
LOSS_WEIGHTS = (0.2, 0.1, 0.2, 0.3, 0.4)


def even_batch_sizes(start: int, stop: int):
    start = int(start)
    stop = int(stop)
    if start <= 0 or stop <= 0 or start % 2 or stop % 2 or start > stop:
        raise ValueError("batch-size bounds must be positive even numbers with start <= stop")
    return tuple(range(start, stop + 1, 2))


def select_largest_stable(results, minimum_headroom_mb: float = 1024.0):
    stable = [
        int(result["batch_size"])
        for result in results
        if result.get("status") == "ok"
        and float(result.get("estimated_headroom_mb", 0.0)) >= minimum_headroom_mb
    ]
    return max(stable) if stable else None


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scaled_size(base_size: int, rate: float) -> int:
    return int(round(base_size * rate / 32) * 32)


def repranet_loss(outputs, masks):
    losses = tuple(structure_loss(output, masks) for output in outputs)
    total = sum(weight * loss for weight, loss in zip(LOSS_WEIGHTS, losses))
    return total, losses


def memory_report(device: torch.device, minimum_observed_free_mb: float):
    divisor = 1024**2
    total_mb = torch.cuda.get_device_properties(device).total_memory / divisor
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / divisor
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / divisor
    free_after_mb, _ = torch.cuda.mem_get_info(device)
    return {
        "total_memory_mb": round(total_mb, 2),
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "estimated_headroom_mb": round(total_mb - peak_reserved_mb, 2),
        "minimum_observed_free_mb": round(minimum_observed_free_mb, 2),
        "free_after_worker_mb": round(free_after_mb / divisor, 2),
    }


def worker_result_base(args, started_at: float):
    return {
        "worker_kind": args.worker_kind,
        "batch_size": args.batchsize,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "precision": "fp32",
        "base_size": args.trainsize,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }


def run_worker(args) -> int:
    started_at = time.perf_counter()
    output_path = Path(args.worker_output)
    if not torch.cuda.is_available():
        result = {
            **worker_result_base(args, started_at),
            "status": "error",
            "error": "CUDA is required for RePraNet3 batch-size probing",
        }
        write_json_atomic(output_path, result)
        return 1

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    minimum_observed_free_mb = float("inf")

    try:
        loader = get_loader(
            image_root=None,
            gt_root=None,
            batchsize=args.batchsize,
            trainsize=args.trainsize,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            split_file=args.split_file,
            dataset_root=args.train_path,
            use_augmentation=True,
        )
        model = RePraNet(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
        ).to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
        )
        total_trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        torch.cuda.reset_peak_memory_stats(device)
        iterator = iter(loader)
        loss_history = []
        output_shapes = None
        rates = (1.25,) if args.worker_kind == "candidate" else SIZE_RATES
        full_batches = 1 if args.worker_kind == "candidate" else args.full_batches

        for batch_index in range(full_batches):
            try:
                images, masks = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, masks = next(iterator)
            if images.shape[0] != args.batchsize:
                raise RuntimeError(
                    f"worker expected batch size {args.batchsize}, got {images.shape[0]}"
                )
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            for rate in rates:
                optimizer.zero_grad(set_to_none=True)
                train_size = scaled_size(args.trainsize, rate)
                if rate != 1.0:
                    step_images = F.interpolate(
                        images,
                        size=(train_size, train_size),
                        mode="bilinear",
                        align_corners=True,
                    )
                    step_masks = F.interpolate(
                        masks,
                        size=(train_size, train_size),
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    step_images = images
                    step_masks = masks

                outputs = model(step_images)
                output_shapes = [list(output.shape) for output in outputs]
                expected_shape = tuple(step_masks.shape)
                if any(tuple(output.shape) != expected_shape for output in outputs):
                    raise RuntimeError(
                        f"prediction/mask shape mismatch: outputs={output_shapes}, "
                        f"mask={list(expected_shape)}"
                    )
                total_loss, component_losses = repranet_loss(outputs, step_masks)
                total_loss.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()
                torch.cuda.synchronize(device)

                free_bytes, _ = torch.cuda.mem_get_info(device)
                minimum_observed_free_mb = min(
                    minimum_observed_free_mb,
                    free_bytes / (1024**2),
                )
                loss_history.append(
                    {
                        "batch_index": batch_index,
                        "rate": rate,
                        "train_size": train_size,
                        "total_loss": float(total_loss.detach().cpu().item()),
                        "component_losses": [
                            float(loss.detach().cpu().item()) for loss in component_losses
                        ],
                    }
                )

        result = {
            **worker_result_base(args, started_at),
            "status": "ok",
            "full_batches": full_batches,
            "optimizer_steps": len(loss_history),
            "size_rates": list(rates),
            "total_trainable_parameters": total_trainable,
            "lora_parameters": model.lora_parameter_count,
            "output_shapes_last_step": output_shapes,
            "loss_history": loss_history,
            **memory_report(device, minimum_observed_free_mb),
        }
        write_json_atomic(output_path, result)
        return 0
    except Exception as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )
        result = {
            **worker_result_base(args, started_at),
            "status": "oom" if is_oom else "error",
            "error": repr(error),
        }
        if torch.cuda.is_available():
            try:
                total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                result.update(
                    {
                        "total_memory_mb": round(total_mb, 2),
                        "peak_allocated_mb": round(
                            torch.cuda.max_memory_allocated(device) / (1024**2), 2
                        ),
                        "peak_reserved_mb": round(
                            torch.cuda.max_memory_reserved(device) / (1024**2), 2
                        ),
                        "estimated_headroom_mb": round(
                            total_mb
                            - torch.cuda.max_memory_reserved(device) / (1024**2),
                            2,
                        ),
                    }
                )
            except Exception:
                pass
        write_json_atomic(output_path, result)
        return 0 if is_oom else 1


def launch_worker(args, batch_size: int, worker_kind: str, output_path: Path, log_path: Path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-kind",
        worker_kind,
        "--worker-output",
        str(output_path),
        "--batchsize",
        str(batch_size),
        "--trainsize",
        str(args.trainsize),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--train-path",
        args.train_path,
        "--split-file",
        args.split_file,
        "--num-workers",
        str(args.num_workers),
        "--full-batches",
        str(args.full_batches),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    else:
        result = {
            "worker_kind": worker_kind,
            "batch_size": batch_size,
            "status": "error",
            "error": f"worker exited {completed.returncode} without a JSON result",
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
    result["worker_returncode"] = completed.returncode
    result["worker_log"] = str(log_path.resolve())
    return result


def run_parent(args) -> int:
    output_path = Path(args.output).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    worker_dir = PROJECT_ROOT / "background_logs" / f"batch_probe_{timestamp}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now().isoformat(),
        "config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "precision": "fp32",
            "lr": args.lr,
            "base_size": args.trainsize,
            "candidate_rate": 1.25,
            "candidate_size": scaled_size(args.trainsize, 1.25),
            "minimum_headroom_mb": args.minimum_headroom_mb,
            "target_headroom_mb": [1024, 3072],
            "full_validation_batches": args.full_batches,
            "full_validation_steps": args.full_batches * len(SIZE_RATES),
            "train_path": args.train_path,
            "split_file": args.split_file,
        },
        "worker_dir": str(worker_dir.resolve()),
        "candidates": [],
        "selected_batch_size": None,
        "final_validation": None,
    }
    write_json_atomic(output_path, report)

    for batch_size in even_batch_sizes(args.start_batch_size, args.max_batch_size):
        print(f"[Batch probe] candidate={batch_size} size={scaled_size(args.trainsize, 1.25)}")
        result = launch_worker(
            args,
            batch_size,
            "candidate",
            worker_dir / f"candidate_bs{batch_size}.json",
            worker_dir / f"candidate_bs{batch_size}.log",
        )
        report["candidates"].append(result)
        write_json_atomic(output_path, report)
        print(
            "[Batch probe] candidate={} status={} peak_reserved_mb={} headroom_mb={}".format(
                batch_size,
                result.get("status"),
                result.get("peak_reserved_mb"),
                result.get("estimated_headroom_mb"),
            )
        )
        if result.get("status") != "ok":
            break
        if float(result.get("estimated_headroom_mb", 0.0)) < args.minimum_headroom_mb:
            break

    selected = select_largest_stable(report["candidates"], args.minimum_headroom_mb)
    report["selected_batch_size"] = selected
    if selected is None:
        report["status"] = "failed_no_stable_batch_size"
        report["finished_at"] = datetime.now().isoformat()
        write_json_atomic(output_path, report)
        print("[Batch probe] batch size 2 was not stable; formal training must not start")
        return 2

    print(
        f"[Batch probe] validating batch={selected} for "
        f"{args.full_batches} full multi-scale batches"
    )
    final_validation = launch_worker(
        args,
        selected,
        "validation",
        worker_dir / f"validation_bs{selected}.json",
        worker_dir / f"validation_bs{selected}.log",
    )
    report["final_validation"] = final_validation
    report["status"] = (
        "ok" if final_validation.get("status") == "ok" else "failed_final_validation"
    )
    report["finished_at"] = datetime.now().isoformat()
    write_json_atomic(output_path, report)
    print(
        "[Batch probe] selected={} final_status={} peak_reserved_mb={} headroom_mb={}".format(
            selected,
            final_validation.get("status"),
            final_validation.get("peak_reserved_mb"),
            final_validation.get("estimated_headroom_mb"),
        )
    )
    print(f"[Batch probe] report={output_path}")
    return 0 if final_validation.get("status") == "ok" else 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-kind",
        choices=("candidate", "validation"),
        default="candidate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    parser.add_argument("--batchsize", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--start-batch-size", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--minimum-headroom-mb", type=float, default=1024.0)
    parser.add_argument("--full-batches", type=int, default=3)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--lora-rank", type=int, default=32, choices=(8, 16, 32, 64, 128))
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-path",
        default="/HDD/pengzhipeng/dataset/TrainDataset",
    )
    parser.add_argument(
        "--split-file",
        default=str(PROJECT_ROOT / "utils" / "TrainDataset" / "train.txt"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "batch_size_probe_rank32_fp32.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.full_batches <= 0:
        raise ValueError("--full-batches must be positive")
    if args.minimum_headroom_mb <= 0:
        raise ValueError("--minimum-headroom-mb must be positive")
    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        if args.batchsize <= 0 or args.batchsize % 2:
            raise ValueError("--batchsize must be a positive even number")
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
