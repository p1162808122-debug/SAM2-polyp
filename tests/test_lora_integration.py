import unittest

import torch
import torch.nn as nn

from lib.RePraNet import RePraNet
from lib.models.lora import LoRALinear
from tools.validate_lora_ranks import build_rank_plan


class LoRAIntegrationTest(unittest.TestCase):
    def test_lora_linear_initializes_as_identity(self):
        torch.manual_seed(2026)
        base = nn.Linear(12, 20)
        layer = LoRALinear(base, rank=8, alpha=16, dropout=0.0)
        x = torch.randn(3, 12)

        self.assertTrue(torch.allclose(layer(x), base(x), atol=1e-6, rtol=1e-6))
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)

    def test_repranet_injects_all_hiera_attention_targets(self):
        model = RePraNet(lora_rank=8, lora_alpha=16)

        self.assertEqual(len(model.lora_target_names), 96)
        self.assertEqual(model.lora_parameter_count, 1305216)
        self.assertTrue(
            all(name.endswith((".attn.qkv", ".attn.proj")) for name in model.lora_target_names)
        )

    def test_lora_receives_gradient_while_frozen_sam2_does_not(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RePraNet(lora_rank=8, lora_alpha=16).to(device)
        model.train()

        x = torch.randn(1, 3, 352, 352, device=device)
        outputs = model(x)
        loss = sum(output.mean() for output in outputs)
        loss.backward()

        lora_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ".lora_A" in name or ".lora_B" in name
        ]
        self.assertTrue(lora_grads)
        self.assertTrue(
            any(gradient is not None and gradient.abs().sum() > 0 for gradient in lora_grads)
        )

        frozen_base = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("sam2.") and ".lora_" not in name
        ]
        self.assertTrue(frozen_base)
        self.assertTrue(all(parameter.requires_grad is False for parameter in frozen_base))
        self.assertTrue(all(parameter.grad is None for parameter in frozen_base))

    def test_rank_plan_uses_consistent_alpha_scaling(self):
        self.assertEqual(
            build_rank_plan((8, 16, 32, 64)),
            ((8, 16.0), (16, 32.0), (32, 64.0), (64, 128.0)),
        )

    def test_rank64_injects_expected_lora_parameters(self):
        model = RePraNet(lora_rank=64, lora_alpha=128)

        self.assertEqual(len(model.lora_target_names), 96)
        self.assertEqual(model.lora_parameter_count, 10441728)

    def test_rank_change_does_not_change_non_lora_initialization(self):
        torch.manual_seed(2026)
        rank8 = RePraNet(lora_rank=8, lora_alpha=16)
        torch.manual_seed(2026)
        rank16 = RePraNet(lora_rank=16, lora_alpha=32)

        rank8_parameters = {
            name: parameter
            for name, parameter in rank8.named_parameters()
            if not name.startswith("sam2.")
        }
        rank16_parameters = {
            name: parameter
            for name, parameter in rank16.named_parameters()
            if not name.startswith("sam2.")
        }
        self.assertEqual(rank8_parameters.keys(), rank16_parameters.keys())
        for name in rank8_parameters:
            self.assertTrue(torch.allclose(rank8_parameters[name], rank16_parameters[name]))


if __name__ == "__main__":
    unittest.main()
