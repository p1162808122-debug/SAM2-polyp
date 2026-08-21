import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.AMGA import DualGateFusion


class DualGateFusionTest(unittest.TestCase):
    def test_output_shape_is_preserved_for_all_pyramid_channels(self):
        for channels, spatial_size in ((64, 16), (128, 8), (256, 4), (512, 2)):
            with self.subTest(channels=channels):
                module = DualGateFusion(channels)
                sam2 = torch.randn(2, channels, spatial_size, spatial_size)
                overlock = torch.randn_like(sam2)

                output = module(sam2, overlock)

                self.assertEqual(tuple(output.shape), tuple(sam2.shape))

    def test_inputs_gates_and_refine_layer_receive_gradients(self):
        module = DualGateFusion(64)
        sam2 = torch.randn(2, 64, 8, 8, requires_grad=True)
        overlock = torch.randn(2, 64, 8, 8, requires_grad=True)

        module(sam2, overlock).square().mean().backward()

        self.assertIsNotNone(sam2.grad)
        self.assertGreater(sam2.grad.abs().sum().item(), 0)
        self.assertIsNotNone(overlock.grad)
        self.assertGreater(overlock.grad.abs().sum().item(), 0)
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_spatial_mismatch_raises_clear_error(self):
        module = DualGateFusion(64)
        sam2 = torch.randn(1, 64, 8, 8)
        overlock = torch.randn(1, 64, 7, 8)

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            module(sam2, overlock)

    def test_channel_mismatch_raises_clear_error(self):
        module = DualGateFusion(64)
        sam2 = torch.randn(1, 64, 8, 8)
        overlock = torch.randn(1, 32, 8, 8)

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            module(sam2, overlock)


if __name__ == "__main__":
    unittest.main()
