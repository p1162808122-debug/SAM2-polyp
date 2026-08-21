import torch
from thop import profile
import time
from lib.RePraNet import RePraNet


def compute_model_stats(testsize=352, device=None, n_runs=50, warmup=10):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    model = RePraNet().to(device)
    model.eval()

    # ── Params ────────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters())

    # ── FLOPs ────────────────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 3, testsize, testsize).to(device)
    with torch.no_grad():
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    # ── FPS ─────────────────────────────────────────────────────────────────
    # Use the model's primary output (lateral_map_1) for timing consistency
    with torch.no_grad():
        for _ in range(warmup):
            out = model(dummy_input)
            if isinstance(out, tuple):
                _ = out[-1]  # use the last output as the primary result
            else:
                _ = out
        if is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            out = model(dummy_input)
            if isinstance(out, tuple):
                _ = out[-1]
            else:
                _ = out
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

    fps = n_runs / (end - start)
    return total_params, trainable_params, flops, fps


if __name__ == "__main__":
    params, train_params, flops, fps = compute_model_stats()

    print("=" * 50)
    print(f"Total Params     : {params:,}")
    print(f"Trainable Params : {train_params:,}")
    print(f"FLOPs            : {flops / 1e9:.2f} G")
    print(f"FPS              : {fps:.1f}")
    print("=" * 50)
