"""
drex.models.transformer — DrexConfig, DrexLayer, DrexTransformer.

Each layer has HybridAttention (L1+L2) + FeedForward.
Model carries a list of LayerState across segment boundaries.

Phase 3 (L3): when config.use_l3=True, DrexTransformer creates TitanMemory
instances and an L3MemoryBridge; each DrexLayer holds a reference to the bridge
and writes mean-pooled representations to disk after every segment step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drex.models.attention import HybridAttention
from drex.models.memory import LayerState, MemoryModule

if TYPE_CHECKING:
    from drex.models.memory import L3MemoryBridge


@dataclass
class DrexConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    window_size: int = 2048
    ff_mult: int = 4           # feed-forward hidden = d_model * ff_mult
    vocab_size: int = 32_000
    max_seq_len: int = 4096
    dropout: float = 0.0
    gradient_checkpointing: bool = False  # recompute activations in backward to save memory
    use_l3: bool = False       # enable L3 disk cache (requires Rust extension)
    l3_base_path: str = "/tmp/drex_l3"
    l3_compress: bool = False
    use_episodic_memory: bool = False   # enable MemoryModule per layer (Phase 13)
    episodic_gate_thresh: float = 0.70  # OR-gate threshold (exp_48_1, Phase 12)
    # Ablation flags (Phase 16 — §12.2 medium-confidence components)
    use_null_gate: bool = True          # null retrieval gate in MemoryModule
    full_seq_residual: bool = False     # apply memory residual to all positions (default: last only)
    memory_last_layer_only: bool = False  # restrict MemoryModule to the final layer only


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DrexLayer(nn.Module):
    def __init__(
        self,
        config: DrexConfig,
        layer_idx: int = 0,
        l3_bridge: Optional["L3MemoryBridge"] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.l3_bridge = l3_bridge

        self.attn = HybridAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            window_size=config.window_size,
        )
        self.ff = FeedForward(config.d_model, config.ff_mult, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Episodic/semantic memory module (optional, Phase 13).
        # When memory_last_layer_only=True, only the last layer (layer_idx == n_layers-1)
        # gets a MemoryModule — all other layers behave as baseline.
        _this_layer_has_mem = (
            config.use_episodic_memory
            and (
                not config.memory_last_layer_only
                or layer_idx == config.n_layers - 1
            )
        )
        self._full_seq_residual: bool = config.full_seq_residual
        self.episodic_mem: Optional[MemoryModule] = (
            MemoryModule(
                config.d_model,
                gate_thresh=config.episodic_gate_thresh,
                use_null_gate=config.use_null_gate,
            )
            if _this_layer_has_mem
            else None
        )
        self.norm_mem: Optional[nn.LayerNorm] = (
            nn.LayerNorm(config.d_model)
            if _this_layer_has_mem
            else None
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, S, d_model)
        layer_state: LayerState,
    ) -> tuple[torch.Tensor, LayerState]:
        # Pre-norm + residual for attention
        normed = self.norm1(x)
        attn_out, new_memory = self.attn(normed, layer_state.memory)
        x = x + attn_out

        # Pre-norm + residual for feed-forward
        x = x + self.ff(self.norm2(x))

        # Episodic/semantic memory: read from accumulated context and inject as residual.
        # Default: add retrieval only at the last (query) token position.
        # full_seq_residual=True: broadcast the same retrieval vector to all positions.
        if self.episodic_mem is not None and self.norm_mem is not None:
            mem_r = self.episodic_mem(self.norm_mem(x))   # (B, d_model)
            if self._full_seq_residual:
                x = x + mem_r.unsqueeze(1)                # broadcast → (B, S, d_model)
            else:
                x = x.clone()
                x[:, -1] = x[:, -1] + mem_r

        new_state = LayerState(memory=new_memory, step=layer_state.step + 1)

        # L3: write representative (mean-pooled over seq and batch[0]) to disk
        if self.l3_bridge is not None:
            rep = x.detach().mean(dim=1)[0].cpu()  # (d_model,) on CPU
            self.l3_bridge.write_and_snapshot(
                layer=self.layer_idx,
                head=0,
                step=layer_state.step,
                key_vec=rep,
                value_vec=rep,
            )
            self.l3_bridge.trigger_prefetch(
                layer=self.layer_idx,
                query_vec=rep,
                k=4,
            )

        return x, new_state


class DrexTransformer(nn.Module):
    """
    Full Drex model.

    For training over long sequences, call forward() on each segment and
    thread states through. Use state.detach() at segment boundaries for TBPTT.

    When config.use_l3=True, each layer writes to disk via L3MemoryBridge and
    fires async prefetch. TitanMemory instances are stored as a plain Python
    list (not nn.ModuleList) so that model.to(device) leaves them on CPU.
    """

    def __init__(self, config: DrexConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # L3: create TitanMemory + bridge before layers (bridge passed into each layer)
        if config.use_l3:
            from drex.models.memory import L3MemoryBridge, TitanMemory

            # Plain list — not nn.ModuleList so model.to() won't move them to GPU.
            # TitanMemory always runs CPU; inputs are .cpu()'d before writes.
            self._titan_list: Optional[list] = [
                TitanMemory(config.d_model, config.d_model * 2)
                for _ in range(config.n_layers)
            ]
            self._l3_bridge: Optional[L3MemoryBridge] = L3MemoryBridge(
                self._titan_list,
                base_path=config.l3_base_path,
                compress=config.l3_compress,
            )
            self.layers = nn.ModuleList([
                DrexLayer(config, i, self._l3_bridge)
                for i in range(config.n_layers)
            ])
        else:
            self._titan_list = None
            self._l3_bridge = None
            self.layers = nn.ModuleList([
                DrexLayer(config, i)
                for i in range(config.n_layers)
            ])

        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding and LM head weights
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def init_states(self, batch: int, device: torch.device) -> list[LayerState]:
        """Create fresh zero states for all layers."""
        cfg = self.config
        d_k = cfg.d_model // cfg.n_heads
        return [
            LayerState.zeros(batch, cfg.n_heads, d_k, d_k, device)
            for _ in range(cfg.n_layers)
        ]

    @staticmethod
    def _ckpt_forward(layer: DrexLayer, step: int, x: torch.Tensor, M: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Checkpoint-compatible layer call: unpacks/repacks LayerState as tensors."""
        from drex.models.memory import LayerState, MemoryState
        state = LayerState(memory=MemoryState(M=M, z=z), step=step)
        x_out, new_state = layer(x, state)
        return x_out, new_state.memory.M, new_state.memory.z

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, S)
        states: Optional[list[LayerState]] = None,
    ) -> tuple[torch.Tensor, list[LayerState]]:
        """
        Returns:
            logits: (B, S, vocab_size)
            new_states: list of LayerState, one per layer
        """
        B, S = input_ids.shape
        device = input_ids.device

        if states is None:
            states = self.init_states(B, device)

        # Positions: 0..S-1 relative to segment start
        pos = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        use_ckpt = self.config.gradient_checkpointing and self.training
        new_states: list[LayerState] = []
        for layer, state in zip(self.layers, states):
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                from drex.models.memory import LayerState, MemoryState
                x, new_M, new_z = checkpoint(
                    DrexTransformer._ckpt_forward,
                    layer, state.step, x, state.memory.M, state.memory.z,
                    use_reentrant=False,
                )
                new_state = LayerState(
                    memory=MemoryState(M=new_M, z=new_z),
                    step=state.step + 1,
                )
            else:
                x, new_state = layer(x, state)
            new_states.append(new_state)

        logits = self.lm_head(self.norm_out(x))  # (B, S, vocab_size)
        return logits, new_states
