"""
Sentinel LLM — Custom Transformer Architecture
Built entirely from scratch with PyTorch.

Designed to run on an NVIDIA RTX 3050 (4 GB VRAM).
Architecture: GPT-2 style pre-norm transformer with dual heads:
  1. Intent classification head  (classify user commands)
  2. Language-model head         (generative responses, tied weights)

All hyperparameters live in ModelConfig — swap them to scale up.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Vocabulary / sequence
    vocab_size:    int   = 8192
    max_seq_len:   int   = 128

    # Architecture
    d_model:       int   = 256     # embedding dimension
    n_heads:       int   = 8       # attention heads
    n_layers:      int   = 6       # transformer blocks
    d_ff:          int   = 1024    # feed-forward hidden dim
    dropout:       float = 0.10

    # Task heads
    num_intents:   int   = 25      # classification output classes

    # Special tokens
    pad_token_id:  int   = 0
    unk_token_id:  int   = 1
    cls_token_id:  int   = 2
    sep_token_id:  int   = 3

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


# ─────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention (bidirectional).
    Uses separate Q / K / V projections; no bias on QKV (GPT-2 style).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k     = config.d_model // config.n_heads
        self.scale   = math.sqrt(self.d_k)

        self.q_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, C)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=pad
    ) -> torch.Tensor:
        B, T, C = x.shape

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = _split(self.q_proj(x)), _split(self.k_proj(x)), _split(self.v_proj(x))

        # (B, heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            # Broadcast over heads and query positions
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)                     # (B, heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ─────────────────────────────────────────────
# Feed-Forward
# ─────────────────────────────────────────────

class PositionWiseFeedForward(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block (GPT-2 / PaLM style).
    Residual connections before normalisation for better gradient flow.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ff   = PositionWiseFeedForward(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# ─────────────────────────────────────────────
# Pooling utilities
# ─────────────────────────────────────────────

def _masked_mean_pool(
    hidden: torch.Tensor,                    # (B, T, C)
    key_padding_mask: Optional[torch.Tensor] # (B, T) True=pad
) -> torch.Tensor:                           # (B, C)
    """Mean-pool over real (non-padding) tokens."""
    if key_padding_mask is None:
        return hidden.mean(dim=1)
    mask = (~key_padding_mask).float().unsqueeze(-1)   # (B, T, 1)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────

class SentinelTransformer(nn.Module):
    """
    Sentinel's core language model.

    Usage modes
    -----------
    classify(input_ids, mask) -> intent logits (B, num_intents)
    lm_logits(input_ids)      -> token logits  (B, T, vocab_size)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Embeddings ──────────────────────────────────────────────
        self.token_emb = nn.Embedding(
            config.vocab_size, config.d_model,
            padding_idx=config.pad_token_id
        )
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        # ── Transformer stack ────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)

        # ── Intent classification head ───────────────────────────────
        self.intent_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_intents),
        )

        # ── LM head (weight-tied to token embeddings) ────────────────
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        # ── Weight initialisation ────────────────────────────────────
        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    # ── Initialisation ──────────────────────────────────────────────

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Shared encoder ───────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,                         # (B, T)
        key_padding_mask: Optional[torch.Tensor] = None, # (B, T)
    ) -> torch.Tensor:                                   # (B, T, d_model)
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x, key_padding_mask)

        return self.ln_f(x)

    # ── Task heads ───────────────────────────────────────────────────

    def classify(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                                   # (B, num_intents)
        hidden = self.encode(input_ids, key_padding_mask)
        pooled = _masked_mean_pool(hidden, key_padding_mask)
        return self.intent_head(pooled)

    def lm_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:                                   # (B, T, vocab_size)
        hidden = self.encode(input_ids)
        return self.lm_head(hidden)

    # ── Convenience ─────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total = self.count_parameters()
        emb   = sum(p.numel() for p in self.token_emb.parameters())
        attn  = sum(
            p.numel()
            for blk in self.blocks
            for p in blk.attn.parameters()
        )
        ff    = sum(
            p.numel()
            for blk in self.blocks
            for p in blk.ff.parameters()
        )
        return (
            f"SentinelTransformer  |  {total/1e6:.2f}M parameters\n"
            f"  Embeddings : {emb/1e6:.2f}M\n"
            f"  Attention  : {attn/1e6:.2f}M\n"
            f"  FeedForward: {ff/1e6:.2f}M\n"
            f"  Other      : {(total-emb-attn-ff)/1e6:.2f}M\n"
            f"  d_model={self.config.d_model}, "
            f"n_layers={self.config.n_layers}, "
            f"n_heads={self.config.n_heads}"
        )

    def estimated_vram_mb(self) -> float:
        """Rough VRAM estimate for fp16 inference."""
        return self.count_parameters() * 2 / 1_048_576   # 2 bytes per fp16 param


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_sentinel_model(num_intents: int = 25, vocab_size: int = 8192, config=None) -> "SentinelTransformer":
    """Return a model instance with the recommended demo config.

    Parameters
    ----------
    num_intents : int
        Number of intent classes for the classification head.
    vocab_size : int
        BPE vocabulary size.
    config : ModelConfig | None
        If provided, overrides default config (num_intents and vocab_size are
        injected into it).  Useful for smoke-tests or custom architectures.
    """
    if config is not None:
        config.num_intents = num_intents
        config.vocab_size  = vocab_size
        return SentinelTransformer(config)

    cfg = ModelConfig(
        vocab_size   = vocab_size,
        num_intents  = num_intents,
        d_model      = 256,
        n_heads      = 8,
        n_layers     = 6,
        d_ff         = 1024,
        max_seq_len  = 128,
        dropout      = 0.10,
    )
    return SentinelTransformer(cfg)
