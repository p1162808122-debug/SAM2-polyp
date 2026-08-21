"""LoRA adapters and injection helpers for the RePraNet2 SAM2 trunk."""

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a low-rank trainable residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(2 * rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        update = F.linear(self.lora_dropout(x), self.lora_A)
        update = F.linear(update, self.lora_B)
        return base_output + self.scaling * update

    @property
    def trainable_parameter_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def _resolve_child(module: nn.Module, name: str) -> nn.Module:
    if name.isdigit():
        return module[int(name)]  # type: ignore[index]
    return getattr(module, name)


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for name in parts[:-1]:
        parent = _resolve_child(parent, name)
    return parent, parts[-1]


def _replace_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if name.isdigit():
        parent[int(name)] = child  # type: ignore[index]
    else:
        setattr(parent, name, child)


def inject_lora_into_hiera(
    trunk: nn.Module,
    rank: int,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_blocks: Optional[Sequence[int]] = None,
) -> Tuple[str, ...]:
    """Inject LoRA into Hiera attention qkv and output projections."""

    blocks = list(range(len(trunk.blocks))) if target_blocks is None else list(target_blocks)
    target_names = []
    for block_idx in blocks:
        if block_idx < 0 or block_idx >= len(trunk.blocks):
            raise IndexError(f"Hiera block index out of range: {block_idx}")
        for suffix in ("attn.qkv", "attn.proj"):
            name = f"blocks.{block_idx}.{suffix}"
            parent, child_name = _resolve_parent(trunk, name)
            base = _resolve_child(parent, child_name)
            if isinstance(base, LoRALinear):
                raise ValueError(f"LoRA is already injected at {name}")
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {name}, got {type(base).__name__}")
            _replace_child(parent, child_name, LoRALinear(base, rank, alpha, dropout))
            target_names.append(name)
    return tuple(target_names)


def iter_lora_modules(module: nn.Module) -> Iterable[LoRALinear]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def count_lora_parameters(module: nn.Module) -> int:
    return sum(adapter.trainable_parameter_count for adapter in iter_lora_modules(module))


def set_lora_train_mode(module: nn.Module, mode: bool) -> None:
    for adapter in iter_lora_modules(module):
        adapter.train(mode)
