import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_HEAD_KEYS = {
    "head.0.weight",
    "head.1.weight",
    "head.1.bias",
    "head.1.running_mean",
    "head.1.running_var",
    "head.4.weight",
    "head.4.bias",
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by OverLoCK iGEMM")
class DualBackboneIntegrationTest(unittest.TestCase):
    def test_dual_backbone_shapes_wiring_outputs_and_gradients(self):
        from lib.RePraNet import RePraNet

        torch.manual_seed(2026)
        model = RePraNet(lora_rank=32, lora_alpha=64).cuda().train()

        self.assertEqual(set(model.overlock_missing_keys), EXPECTED_HEAD_KEYS)
        self.assertEqual(tuple(model.overlock_unexpected_keys), ())
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))

        projection_outputs = []
        fusion_outputs = []
        rfb_inputs = []
        handles = []
        for projection in (
            model.sam2_proj1,
            model.sam2_proj2,
            model.sam2_proj3,
            model.sam2_proj4,
        ):
            handles.append(
                projection.register_forward_hook(
                    lambda _module, _inputs, output: projection_outputs.append(output)
                )
            )
        for fusion in (model.fusion1, model.fusion2, model.fusion3, model.fusion4):
            handles.append(
                fusion.register_forward_hook(
                    lambda _module, _inputs, output: fusion_outputs.append(output)
                )
            )
        for rfb in (model.rfb1_1, model.rfb2_1, model.rfb3_1, model.rfb4_1):
            handles.append(
                rfb.register_forward_pre_hook(
                    lambda _module, inputs: rfb_inputs.append(inputs[0])
                )
            )

        try:
            image = torch.randn(1, 3, 352, 352, device="cuda")
            outputs = model(image)
        finally:
            for handle in handles:
                handle.remove()

        expected_shapes = [
            (1, 64, 88, 88),
            (1, 128, 44, 44),
            (1, 256, 22, 22),
            (1, 512, 11, 11),
        ]
        self.assertEqual([tuple(tensor.shape) for tensor in projection_outputs], expected_shapes)
        self.assertEqual([tuple(tensor.shape) for tensor in fusion_outputs], expected_shapes)
        self.assertEqual(len(rfb_inputs), 4)
        self.assertTrue(
            all(fused is rfb_input for fused, rfb_input in zip(fusion_outputs, rfb_inputs))
        )
        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(tuple(output.shape) == (1, 1, 352, 352) for output in outputs))

        sum(output.mean() for output in outputs).backward()

        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for name, parameter in model.backbone.named_parameters()
                if not name.startswith("head.")
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for name, parameter in model.sam2.named_parameters()
                if ".lora_" in name
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for projection in (
                    model.sam2_proj1,
                    model.sam2_proj2,
                    model.sam2_proj3,
                    model.sam2_proj4,
                )
                for parameter in projection.parameters()
            )
        )
        for fusion in (model.fusion1, model.fusion2, model.fusion3, model.fusion4):
            self.assertTrue(
                all(
                    parameter.grad is not None and parameter.grad.abs().sum() > 0
                    for parameter in fusion.parameters()
                )
            )
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in model.ra1_conv2.parameters()
            )
        )

        frozen_sam2 = [
            parameter
            for name, parameter in model.sam2.named_parameters()
            if ".lora_" not in name
        ]
        self.assertTrue(frozen_sam2)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen_sam2))
        self.assertTrue(all(parameter.grad is None for parameter in frozen_sam2))


if __name__ == "__main__":
    unittest.main()
