import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

EPS = 1e-12


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
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p

    raise FileNotFoundError(f"GT not found for prediction: {pred_name} under {gt_dir}")


def fmeasure_calu(smap: np.ndarray, gt_map: np.ndarray, threshold: float):
    if threshold > 1:
        threshold = 1

    label = np.zeros_like(smap, dtype=np.uint8)
    label[smap >= threshold] = 1

    num_rec = np.count_nonzero(label == 1)
    num_no_rec = np.count_nonzero(label == 0)
    label_and = np.logical_and(label == 1, gt_map == 1)
    num_and = np.count_nonzero(label_and)
    num_obj = np.sum(gt_map)
    num_pred = np.sum(label)

    fn = num_obj - num_and
    fp = num_rec - num_and
    tn = num_no_rec - fn

    if num_and == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    iou = num_and / (fn + num_rec + EPS)
    precision = num_and / (num_rec + EPS)
    recall = num_and / (num_obj + EPS)
    specificity = tn / (tn + fp + EPS)
    dice = 2.0 * num_and / (num_obj + num_pred + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    return precision, recall, specificity, dice, f1, iou


def alignment_term(dfm: np.ndarray, dgt: np.ndarray) -> np.ndarray:
    mu_fm = np.mean(dfm)
    mu_gt = np.mean(dgt)
    align_fm = dfm - mu_fm
    align_gt = dgt - mu_gt
    return 2.0 * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + EPS)


def enhanced_alignment_term(align_matrix: np.ndarray) -> np.ndarray:
    return ((align_matrix + 1.0) ** 2) / 4.0


def enhanced_measure(fm: np.ndarray, gt: np.ndarray) -> float:
    fm = fm.astype(bool)
    gt = gt.astype(bool)
    dfm = fm.astype(np.float64)
    dgt = gt.astype(np.float64)

    if np.sum(dgt) == 0:
        enhanced = 1.0 - dfm
    elif np.sum(~gt) == 0:
        enhanced = dfm
    else:
        align = alignment_term(dfm, dgt)
        enhanced = enhanced_alignment_term(align)

    h, w = gt.shape
    return float(np.sum(enhanced) / (h * w - 1 + EPS))


def object_score(prediction: np.ndarray, gt: np.ndarray) -> float:
    if prediction.size == 0:
        return 0.0

    vals = prediction[gt]
    if vals.size == 0:
        return 0.0

    x = np.mean(vals)
    sigma_x = np.std(vals)
    return float(2.0 * x / (x * x + 1.0 + sigma_x + EPS))


def s_object(prediction: np.ndarray, gt: np.ndarray) -> float:
    prediction_fg = prediction.copy()
    prediction_fg[~gt] = 0
    o_fg = object_score(prediction_fg, gt)

    prediction_bg = 1.0 - prediction
    prediction_bg[gt] = 0
    o_bg = object_score(prediction_bg, ~gt)

    u = np.mean(gt.astype(np.float64))
    return float(u * o_fg + (1.0 - u) * o_bg)


def centroid(gt: np.ndarray):
    rows, cols = gt.shape
    if np.sum(gt) == 0:
        x = int(round(cols / 2.0))
        y = int(round(rows / 2.0))
        return max(1, x), max(1, y)

    total = np.sum(gt)
    i = np.arange(1, cols + 1)
    j = np.arange(1, rows + 1)
    x = int(round(np.sum(np.sum(gt, axis=0) * i) / total))
    y = int(round(np.sum(np.sum(gt, axis=1) * j) / total))
    return max(1, x), max(1, y)


def divide_gt(gt: np.ndarray, x: int, y: int):
    hei, wid = gt.shape
    area = float(wid * hei)

    lt = gt[:y, :x]
    rt = gt[:y, x:wid]
    lb = gt[y:hei, :x]
    rb = gt[y:hei, x:wid]

    w1 = (x * y) / area
    w2 = ((wid - x) * y) / area
    w3 = (x * (hei - y)) / area
    w4 = 1.0 - w1 - w2 - w3
    return lt, rt, lb, rb, w1, w2, w3, w4


def divide_prediction(prediction: np.ndarray, x: int, y: int):
    hei, wid = prediction.shape
    lt = prediction[:y, :x]
    rt = prediction[:y, x:wid]
    lb = prediction[y:hei, :x]
    rb = prediction[y:hei, x:wid]
    return lt, rt, lb, rb


def ssim_region(prediction: np.ndarray, gt: np.ndarray) -> float:
    dgt = gt.astype(np.float64)
    hei, wid = prediction.shape
    n = float(wid * hei)

    x = np.mean(prediction)
    y = np.mean(dgt)

    sigma_x2 = np.sum((prediction - x) ** 2) / (n - 1 + EPS)
    sigma_y2 = np.sum((dgt - y) ** 2) / (n - 1 + EPS)
    sigma_xy = np.sum((prediction - x) * (dgt - y)) / (n - 1 + EPS)

    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        return float(alpha / (beta + EPS))
    if alpha == 0 and beta == 0:
        return 1.0
    return 0.0


def s_region(prediction: np.ndarray, gt: np.ndarray) -> float:
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide_gt(gt, x, y)
    p1, p2, p3, p4 = divide_prediction(prediction, x, y)

    q1 = ssim_region(p1, gt1)
    q2 = ssim_region(p2, gt2)
    q3 = ssim_region(p3, gt3)
    q4 = ssim_region(p4, gt4)
    return float(w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4)


def structure_measure(prediction: np.ndarray, gt: np.ndarray) -> float:
    y = np.mean(gt.astype(np.float64))

    if y == 0:
        return float(1.0 - np.mean(prediction))
    if y == 1:
        return float(np.mean(prediction))

    alpha = 0.5
    q = alpha * s_object(prediction, gt) + (1 - alpha) * s_region(prediction, gt)
    return float(max(q, 0.0))


def weighted_fmeasure(fg: np.ndarray, gt: np.ndarray) -> float:
    dgt = gt.astype(np.float64)
    e = np.abs(fg - dgt)

    dst, idx = distance_transform_edt(~gt, return_indices=True)

    et = e.copy()
    bg = ~gt
    et[bg] = et[idx[0, bg], idx[1, bg]]

    ea = gaussian_filter(et, sigma=5, truncate=((7 - 1) / 2) / 5)

    min_e_ea = e.copy()
    fg_mask = gt & (ea < e)
    min_e_ea[fg_mask] = ea[fg_mask]

    b = np.ones_like(fg)
    b[bg] = 2.0 - np.exp(np.log(1 - 0.5) / 5.0 * dst[bg])
    ew = min_e_ea * b

    tpw = np.sum(dgt) - np.sum(ew[gt])
    fpw = np.sum(ew[bg])

    r = 1.0 - np.mean(ew[gt]) if np.any(gt) else 0.0
    p = tpw / (tpw + fpw + EPS)
    q = (2.0 * r * p) / (r + p + EPS)
    return float(q)


def evaluate_one_dataset(pred_dataset_dir: Path, dataset: str, data_path: Path):
    gt_path = data_path / dataset / "masks"
    pred_files = sorted([p for p in pred_dataset_dir.glob("*.png")])
    img_num = len(pred_files)
    if img_num == 0:
        raise RuntimeError(f"No prediction png files found in {pred_dataset_dir}")

    thresholds = np.linspace(1.0, 0.0, 256)

    threshold_dice = np.zeros((img_num, len(thresholds)), dtype=np.float64)
    threshold_iou = np.zeros((img_num, len(thresholds)), dtype=np.float64)

    for i, pred_path in enumerate(pred_files):
        gt_mask = find_gt_path(gt_path, pred_path.name)
        gt_img = read_mask(gt_mask)
        gt = gt_img > 128

        pred_img = read_mask(pred_path)
        if pred_img.shape != gt.shape:
            pred_img = resize_like(pred_img, gt.shape)

        resmap = pred_img.astype(np.float64) / 255.0
        resmap = normalize_map(resmap)

        gt_pixels = np.count_nonzero(gt)
        for t_idx, threshold in enumerate(thresholds):
            pred_binary = resmap >= threshold
            pred_pixels = np.count_nonzero(pred_binary)
            intersection = np.count_nonzero(pred_binary & gt)
            threshold_dice[i, t_idx] = (
                2.0 * intersection / (gt_pixels + pred_pixels + EPS)
            )
            threshold_iou[i, t_idx] = (
                intersection / (gt_pixels + pred_pixels - intersection + EPS)
            )

    col_dic = np.mean(threshold_dice, axis=0)
    col_iou = np.mean(threshold_iou, axis=0)

    result = {}
    result["meanDic"] = float(np.mean(col_dic))
    result["maxDic"] = float(np.max(col_dic))
    result["meanIoU"] = float(np.mean(col_iou))
    result["maxIoU"] = float(np.max(col_iou))

    result["column_Dic"] = col_dic
    result["column_IoU"] = col_iou
    return result


def find_pred_dataset_dir(results_root: Path, run_name: str, model_subfolder: str, dataset: str) -> Path:
    """
    Search order:
      results/run/model_subfolder/dataset
      results/run/model_subfolder/result1/dataset
      results/run/model_subfolder/test1/dataset
    Falls back to results/run/dataset (old layout).
    """
    run_dir = results_root / run_name

    cands = [
        run_dir / model_subfolder / dataset,
        run_dir / model_subfolder / "result1" / dataset,
        run_dir / model_subfolder / "test1" / dataset,
        # backward compatibility: model == run_name means old flat layout
        results_root / model_subfolder / dataset,
        results_root / model_subfolder / "result1" / dataset,
        results_root / model_subfolder / "test1" / dataset,
    ]

    for cand in cands:
        if cand.exists() and cand.is_dir():
            return cand

    return cands[0]


def auto_discover_latest_run(results_root: Path):
    """
    Returns (run_folder_name, [model_subfolder, ...]) for the run with the
    largest run number found under results_root.
    Returns (None, None) if nothing is found.
    """
    import re

    run_pat = re.compile(r"^run(\d+)_", re.IGNORECASE)
    candidates = []

    if not results_root.exists():
        return None, None

    for d in results_root.iterdir():
        if d.is_dir():
            m = run_pat.match(d.name)
            if m:
                run_num = int(m.group(1))
                candidates.append((run_num, d.name))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    run_name = candidates[0][1]
    run_dir = results_root / run_name

    model_pat = re.compile(r"^(?:last_model|best_model\d*)$", re.IGNORECASE)
    model_subfolders = []
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir() and model_pat.match(sub.name):
            model_subfolders.append(sub.name)

    # Fallback: if no best_modelN found, treat the run folder itself as the model
    if not model_subfolders:
        model_subfolders = [run_name]

    return run_name, model_subfolders


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Evaluate PraNet predictions")

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
        help="Run folder name under <project>/results (e.g. 'run1_20epoch'). "
             "If not provided, auto-detects the run with largest runN from results/. "
             "All best_model/best_modelN subfolders inside it will be evaluated.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"],
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

        import re
        model_pat = re.compile(r"^(?:last_model|best_model\d*)$", re.IGNORECASE)
        model_subfolders = []
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and model_pat.match(sub.name):
                model_subfolders.append(sub.name)
        if not model_subfolders:
            model_subfolders = [run_name]
        print(f"[Run] Using run: {run_name}")
        print(f"[Run] Model subfolders: {model_subfolders}")

    # Evaluate each model subfolder independently
    for model_subfolder in model_subfolders:
        out_root = script_dir / "EvaluateResults" / run_name / model_subfolder
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Run={run_name}  Model={model_subfolder}")
        print(f"Prediction root: {results_root}")
        print(f"Output root:     {out_root}")
        print(f"{'='*60}")

        result_blocks = []
        for dataset in args.datasets:
            mat_dir = out_root / f"{dataset}-mat"
            mat_dir.mkdir(parents=True, exist_ok=True)

            pred_dataset_dir = find_pred_dataset_dir(results_root, run_name, model_subfolder, dataset)

            if not pred_dataset_dir.exists():
                print(f"  [Skip] dataset={dataset}: missing {pred_dataset_dir}")
                continue

            print(f"  Evaluating dataset={dataset} ...")

            result = evaluate_one_dataset(pred_dataset_dir, dataset, data_path)

            np.savez(
                mat_dir / f"{model_subfolder}.npz",
                column_Dic=result["column_Dic"],
                column_IoU=result["column_IoU"],
                maxDic=result["maxDic"],
                maxIoU=result["maxIoU"],
                meanIoU=result["meanIoU"],
                meanDic=result["meanDic"],
            )

            block_lines = [
                f"Run     : {run_name}",
                f"Model   : {model_subfolder}",
                f"Dataset : {dataset}",
                f"meanDic : {result['meanDic']:.3f}",
                f"meanIoU : {result['meanIoU']:.3f}",
                f"maxDice : {result['maxDic']:.3f}",
                f"maxIoU  : {result['maxIoU']:.3f}",
            ]

            block = "\n".join(block_lines)
            print(block)
            result_blocks.append(block + "\n" + "-" * 48 + "\n")

        if result_blocks:
            txt_path = out_root / f"{model_subfolder}_result.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("".join(result_blocks))


if __name__ == "__main__":
    main()
