# SAM2-polyp

Research code for a polyp-segmentation study built around **RePraNet3**, a
dual-backbone architecture that combines SAM2 Hiera-L features with an
OverLoCK-T visual backbone and lightweight LoRA adaptation.

## Abstract

Accurate polyp segmentation must preserve both global semantic context and
fine boundary structure, while remaining practical to train on limited
medical-image data. This implementation explores a two-stream design in which
the frozen SAM2 Hiera-L image encoder provides strong semantic features and
OverLoCK-T contributes complementary multi-scale visual representations. SAM2
features are projected to the corresponding OverLoCK scales and fused with
same-scale DualGateFusion modules. The fused features are refined by
multi-scale feature enhancement (MFE), spatial/scale-guided aggregation (SGA),
and cascaded feature-boundary refinement (CFBR). Five deep-supervision outputs
are produced for coarse-to-fine prediction. LoRA modules are injected into the
SAM2 Hiera attention blocks so that rank/alpha trade-offs can be studied
without updating the complete foundation model.

The repository is intended for research reproduction and further ablation
work. It contains model and experiment code only; private datasets, server
split files, checkpoints, prediction masks, evaluation logs, and pretrained
weights are intentionally excluded.

## Architecture

```text
Input
  ├── SAM2 Hiera-L ── 1×1 projections ──┐
  └── OverLoCK-T ───────────────────────┤
                                        └── DualGateFusion (4 scales)
                                             └── MFE
                                                  └── SGA
                                                       └── CFBR cascade
                                                            └── 5 outputs
```

The implementation uses the following main components:

- frozen SAM2 Hiera-L image features with LoRA adaptation;
- OverLoCK-T multi-scale features;
- `DualGateFusion` for cross-backbone fusion;
- `MFE`, `SGA`, and `CFBR` for multi-scale refinement;
- BCE + IoU + edge-aware structure loss with deep supervision;
- validation-based best-checkpoint selection and patience-based early stopping;
- threshold-swept Dice/IoU evaluation with per-image prediction normalization.

## Repository layout

```text
MyTrain.py                 Training entry point
MyTest.py                  Prediction generation
MyEval.py                  Dice/IoU evaluation
lib/RePraNet.py            RePraNet3 model definition
lib/AMGA.py                Fusion and multi-scale aggregation modules
lib/CFBR.py                Cascaded feature-boundary refinement
lib/models/                SAM2, OverLoCK, and LoRA implementations
lib/models/sam2/           SAM2 model/configuration code
utils/                     Dataset loading and transforms
tests/                     Focused integration/unit tests
tools/                     Batch-size and LoRA validation utilities
run_script.sh              Server-oriented train → test → evaluate wrapper
```

## Data layout

Training data should contain paired `images/` and `masks/` directories. The
split-based training path expects:

```text
TrainDataset/
├── images/
├── masks/
└── ...
```

`utils/TrainDataset/train.txt` and `val.txt` should list image/mask pairs
relative to `TrainDataset`. Test data is expected in the form:

```text
TestDataset/
├── CVC-300/{images,masks}/
├── CVC-ClinicDB/{images,masks}/
├── CVC-ColonDB/{images,masks}/
├── ETIS-LaribPolypDB/{images,masks}/
└── Kvasir/{images,masks}/
```

## Pretrained weights

Weights are not stored in this repository. Before running the model, provide
the compatible upstream checkpoints at the locations expected by the current
implementation, or adjust the paths in `lib/RePraNet.py`:

- SAM2 Hiera-L checkpoint: `../pretrained/sam2_hiera_large.pt`
- OverLoCK-T checkpoint: `lib/EncoderW/overlock_t_in1k_224.pth`

Do not commit these files. They are large binary artifacts and may have
separate upstream licenses or redistribution conditions.

## Training, testing, and evaluation

Install a CUDA-enabled PyTorch environment with the dependencies required by
the imported SAM2, OverLoCK, torchvision, imageio, PIL, NumPy, and
augmentation modules. Then run the stages separately so paths and checkpoints
are explicit:

```bash
python MyTrain.py \
  --epoch 50 \
  --batchsize 4 \
  --trainsize 352 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --train-path /path/to/TrainDataset \
  --split-dir ./utils/TrainDataset \
  --use-augmentation \
  --patience 10

python MyTest.py \
  --run-dir ./checkpoint/run1_50epoch \
  --lora-rank 32 \
  --lora-alpha 64 \
  --test-path /path/to/TestDataset

python MyEval.py \
  --data-path /path/to/TestDataset \
  --models run1_50epoch
```

The existing `run_script.sh` preserves the original server-oriented defaults;
for another machine, pass explicit paths as above or adapt that wrapper
locally. The evaluator's formal `meanDic`/`meanIoU` protocol scans 256
thresholds after per-image min-max normalization. A fixed-threshold 0.5
evaluation is a separate protocol and should not be mixed with the formal
table.

## Reproducibility note

The public repository does not contain the private server's data, split files,
checkpoints, prediction masks, or historical logs. Reproduction therefore
requires independently obtaining the datasets and compatible pretrained
weights, recording the exact split, random seed, LoRA rank/alpha, batch size,
and checkpoint used for evaluation.

Some historical exploratory runs added images originating from a test set to
training. Those scores should be treated as exploratory/adaptation results,
not as independent held-out generalization evidence.

## 中文简介

本项目是一个面向息肉分割的 RePraNet3 研究实现，将 SAM2 Hiera-L 与
OverLoCK-T 双主干进行多尺度融合，并在 SAM2 注意力模块中引入 LoRA，结合
DualGateFusion、MFE、SGA 和 CFBR 逐级细化边界与分割结果。仓库只公开模型、
训练、测试和评估代码，不包含数据集、权重、checkpoint、预测结果和训练日志。
