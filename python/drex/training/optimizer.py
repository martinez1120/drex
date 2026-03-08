"""
drex.training.optimizer — parameter-group-aware AdamW and cosine LR schedule.
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    AdamW with parameter-group weight decay.

    LayerNorm parameters and bias terms are excluded from weight decay
    (they are 1-D and decaying them hurts training stability).
    All other learnable parameters receive the full weight_decay.
    """
    decay_names: set[str] = set()
    no_decay_names: set[str] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:  # pragma: no cover
            continue  # pragma: no cover
        if "bias" in name or "norm" in name.lower():
            no_decay_names.add(name)
        else:
            decay_names.add(name)

    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    groups = [
        {
            "params": [param_dict[n] for n in sorted(decay_names)],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[n] for n in sorted(no_decay_names)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(groups, lr=lr, betas=betas)


def cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup over warmup_steps, then cosine decay to min_lr_ratio * base_lr.

    The returned scheduler multiplies the base lr by the factor returned by
    the lambda at each step.
    """

    def _schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, _schedule)
