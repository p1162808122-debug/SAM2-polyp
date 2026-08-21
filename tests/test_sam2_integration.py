import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SAM2RePraNetIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by OverLoCK iGEMM")
    def test_sam2_trunk_and_repranet_outputs_for_352_input(self):
        from lib.RePraNet import RePraNet

        model = RePraNet().cuda()
        model.eval()

        self.assertIsNotNone(model.sam2)
        frozen_base = [
            parameter
            for name, parameter in model.sam2.named_parameters()
            if ".lora_" not in name
        ]
        lora_parameters = [
            parameter
            for name, parameter in model.sam2.named_parameters()
            if ".lora_" in name
        ]
        self.assertTrue(frozen_base)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen_base))
        self.assertTrue(lora_parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in lora_parameters))

        image = torch.randn(1, 3, 352, 352, device="cuda")
        with torch.no_grad():
            stages = model.sam2.image_encoder.trunk(image)
            outputs = model(image)

        expected_stage_shapes = [
            (1, 144, 88, 88),
            (1, 288, 44, 44),
            (1, 576, 22, 22),
            (1, 1152, 11, 11),
        ]
        self.assertEqual([tuple(stage.shape) for stage in stages], expected_stage_shapes)
        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(tuple(output.shape) == (1, 1, 352, 352) for output in outputs))


if __name__ == "__main__":
    unittest.main()
