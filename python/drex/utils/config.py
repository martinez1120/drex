"""
drex.utils.config — safetensors checkpoint save / load utilities.

Saves model weights to a .safetensors file and stores DrexConfig +
global step in a sidecar .json file with the same stem.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from drex.models.transformer import DrexConfig, DrexTransformer


def save_checkpoint(
    model: DrexTransformer,
    path: str | Path,
    step: int = 0,
) -> None:
    """
    Persist model weights and metadata.

    Creates two files:
        ``path``             — safetensors weight file
        ``path.stem + .json`` — config + training step

    The parent directory is created automatically if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    state = {k: v.cpu().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(state, path)

    meta = {"step": step, "config": asdict(model.config)}
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def load_checkpoint(model: DrexTransformer, path: str | Path) -> int:
    """
    Load weights from a safetensors checkpoint into *model* in-place.

    Returns the global training step stored in the sidecar JSON, or 0 if the
    sidecar does not exist.
    """
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(path)
    model.load_state_dict(state, strict=True)

    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
        return int(meta.get("step", 0))
    return 0
