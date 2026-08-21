import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from pathlib import Path
from .models.sam2.build_sam import build_sam2
from .models.lora import count_lora_parameters, inject_lora_into_hiera, set_lora_train_mode
from .models.overlock import overlock_t
from .CFBR import CFBR
from .AMGA import DualGateFusion, MFE, SGA, BasicConv2d, ConvNormAct

class RePraNet(nn.Module):
    EXPECTED_OVERLOCK_HEAD_KEYS = {
        "head.0.weight",
        "head.1.weight",
        "head.1.bias",
        "head.1.running_mean",
        "head.1.running_var",
        "head.4.weight",
        "head.4.bias",
    }

    def __init__(
        self,
        channel=32,
        lora_rank=8,
        lora_alpha=None,
        lora_dropout=0.0,
    ):
        super(RePraNet, self).__init__()
        if lora_rank not in (8, 16, 32, 64, 128):
            raise ValueError(f"lora_rank must be one of 8, 16, 32, 64, 128, got {lora_rank}")
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = project_root.parent / "pretrained" / "sam2_hiera_large.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM2 Hiera-L checkpoint was not found: {checkpoint_path}"
            )

        self.sam2 = build_sam2(
            config_file="sam2_hiera_l",
            ckpt_path=str(checkpoint_path),
            device="cpu",
            mode="eval",
            apply_postprocessing=False,
        )
        for parameter in self.sam2.parameters():
            parameter.requires_grad = False
        self.sam2.eval()

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(2 * lora_rank if lora_alpha is None else lora_alpha)
        self.lora_dropout = float(lora_dropout)

        self.sam2_proj1 = nn.Conv2d(144, 64, 1)
        self.sam2_proj2 = nn.Conv2d(288, 128, 1)
        self.sam2_proj3 = nn.Conv2d(576, 256, 1)
        self.sam2_proj4 = nn.Conv2d(1152, 512, 1)

        self.backbone = overlock_t(pretrained=False)
        overlock_checkpoint_path = (
            project_root / "lib" / "EncoderW" / "overlock_t_in1k_224.pth"
        )
        if not overlock_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"OverLoCK-T checkpoint was not found: {overlock_checkpoint_path}"
            )
        pretrained_dict = torch.load(overlock_checkpoint_path, map_location="cpu")
        if not isinstance(pretrained_dict, dict):
            raise RuntimeError(
                "OverLoCK-T checkpoint must contain a state dictionary, "
                f"got {type(pretrained_dict).__name__}"
            )
        pretrained_dict = {
            key: value
            for key, value in pretrained_dict.items()
            if not key.startswith("head.")
        }
        incompatible = self.backbone.load_state_dict(pretrained_dict, strict=False)
        self.overlock_missing_keys = tuple(sorted(incompatible.missing_keys))
        self.overlock_unexpected_keys = tuple(sorted(incompatible.unexpected_keys))
        if (
            set(self.overlock_missing_keys) != self.EXPECTED_OVERLOCK_HEAD_KEYS
            or self.overlock_unexpected_keys
        ):
            raise RuntimeError(
                "OverLoCK-T checkpoint mismatch after filtering head.*: "
                f"missing={self.overlock_missing_keys}, "
                f"unexpected={self.overlock_unexpected_keys}"
            )

        self.fusion1 = DualGateFusion(64)
        self.fusion2 = DualGateFusion(128)
        self.fusion3 = DualGateFusion(256)
        self.fusion4 = DualGateFusion(512)

        self.rfb1_1 = MFE(64, channel)
        self.rfb2_1 = MFE(128, channel)
        self.rfb3_1 = MFE(256, channel)
        self.rfb4_1 = MFE(512, channel)
        self.focus0 = CFBR(32)
        self.focus1 = CFBR(64)
        self.focus2 = CFBR(128)
        self.focus3 = CFBR(256)
        self.agg1 = SGA(channel)

        self.ra4_conv1 = BasicConv2d(256, 128, kernel_size=1)
        self.ra4_conv2 = Conv2d(128, 1, kernel_size=1)
        self.ra3_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra3_conv2 = Conv2d(64, 1, kernel_size=1)
        self.ra2_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra2_conv2 = Conv2d(32, 1, kernel_size=1)
        self.ra1_conv1 = BasicConv2d(32, 16, kernel_size=1)
        self.ra1_conv2 = Conv2d(16, 1, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        self.gamma3 = nn.Parameter(torch.ones(1))
        self.gamma4 = nn.Parameter(torch.ones(1))

        # Inject after all RePraNet layers are initialized so rank sweeps use
        # identical non-LoRA initialization under the same random seed.
        self.lora_target_names = inject_lora_into_hiera(
            self.sam2.image_encoder.trunk,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self.lora_parameter_count = count_lora_parameters(self.sam2.image_encoder.trunk)

    def train(self, mode=True):
        super().train(mode)
        self.sam2.eval()
        set_lora_train_mode(self.sam2, mode)
        return self

    def forward(self, x):
        [sam2_stage1, sam2_stage2, sam2_stage3, sam2_stage4] = (
            self.sam2.image_encoder.trunk(x)
        )
        [overlock_stage1, overlock_stage2, overlock_stage3, overlock_stage4] = (
            self.backbone.get_pyramid_features(x)
        )

        sam2_stage1 = self.sam2_proj1(sam2_stage1)
        sam2_stage2 = self.sam2_proj2(sam2_stage2)
        sam2_stage3 = self.sam2_proj3(sam2_stage3)
        sam2_stage4 = self.sam2_proj4(sam2_stage4)
        
        x0 = self.fusion1(sam2_stage1, overlock_stage1)
        x1 = self.fusion2(sam2_stage2, overlock_stage2)
        x2 = self.fusion3(sam2_stage3, overlock_stage3)
        x3 = self.fusion4(sam2_stage4, overlock_stage4)
        x0_rfb, x0_parallel = self.rfb1_1(x0)
        x1_rfb, x1_parallel = self.rfb2_1(x1)
        x2_rfb, x2_parallel = self.rfb3_1(x2)
        x3_rfb, x3_parallel = self.rfb4_1(x3)

        ra5_feat = self.agg1(x3_rfb, x2_rfb, x1_rfb, x0_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=4, mode='bilinear', align_corners=True)   

        crop_4 = F.interpolate(ra5_feat, scale_factor=0.125, mode='bilinear',recompute_scale_factor=True)
        x = self.focus3(x3_parallel, x3_parallel, crop_4)
        x = self.ra4_conv1(x)
        ra4_feat = self.ra4_conv2(x)
        x = ra4_feat + self.gamma4*crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')

        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.focus2(x2_parallel, x2_parallel, crop_3)
        x = self.ra3_conv1(x)
        ra3_feat = self.ra3_conv2(x)
        x = ra3_feat + self.gamma3*crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')

        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.focus1(x1_parallel, x1_parallel, crop_2)
        x = self.ra2_conv1(x)
        ra2_feat = self.ra2_conv2(x)
        x = ra2_feat + self.gamma2*crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')

        crop_1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.focus0(x0_parallel, x0_parallel, crop_1)
        x = self.ra1_conv1(x)
        ra1_feat = self.ra1_conv2(x)
        x = ra1_feat + self.gamma1*crop_1
        lateral_map_1 = F.interpolate(x, scale_factor=4, mode='bilinear')

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1

if __name__ == '__main__':
    ras = RePraNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    out = ras(input_tensor)
