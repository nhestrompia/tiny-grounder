"""The grounder: a small listwise transformer over [query ; NULL ; elements]."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from tokenizers import Tokenizer

from grounder.features import N_FEATS
from grounder.tokenizer import PAD_ID, load_tokenizer

NEG_INF = -1e9  # logit of padded elements
_MASK_BIAS = -1e4  # additive attention bias of padded keys


@dataclass
class GrounderConfig:
    vocab: int = 8192
    d: int = 256
    layers: int = 4
    heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.1
    max_q: int = 48
    max_text: int = 16
    max_ctx: int = 24
    n_roles: int = 16
    n_feats: int = N_FEATS

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GrounderConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


class MultiHeadAttention(nn.Module):
    """Plain multi-head self-attention: matmul, additive key-padding bias, softmax."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_bias: Tensor) -> Tensor:
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + key_bias
        att = self.drop(att.softmax(dim=-1))
        return self.out((att @ v).transpose(1, 2).reshape(b, s, -1))


class Block(nn.Module):
    """Pre-norm transformer layer."""

    def __init__(self, cfg: GrounderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d)
        self.attn = MultiHeadAttention(cfg.d, cfg.heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d, cfg.ffn_mult * cfg.d), nn.GELU(), nn.Linear(cfg.ffn_mult * cfg.d, cfg.d)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, key_bias: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), key_bias))
        return x + self.drop(self.ffn(self.norm2(x)))


class Grounder(nn.Module):
    def __init__(self, cfg: GrounderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.tok_emb = nn.Embedding(cfg.vocab, d, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(cfg.max_q, d)
        self.type_emb = nn.Embedding(2, d)  # 0 = query token, 1 = element
        self.role_emb = nn.Embedding(cfg.n_roles, d)
        self.ctx_proj = nn.Linear(d, d)
        self.feat_mlp = nn.Sequential(nn.Linear(cfg.n_feats, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def _masked_mean(self, ids: Tensor) -> Tensor:
        """Mean embedding over non-PAD tokens of the last dim; all-PAD rows give zeros."""
        keep = (ids != PAD_ID).unsqueeze(-1).to(self.tok_emb.weight.dtype)
        return (self.tok_emb(ids) * keep).sum(-2) / keep.sum(-2).clamp(min=1.0)

    def forward(
        self, query_ids: Tensor, el_text_ids: Tensor, el_ctx_ids: Tensor,
        el_role: Tensor, el_feats: Tensor, el_mask: Tensor,
    ) -> Tensor:
        """Logits [B, N+1]; index 0 is NULL ("no element matches"), padded elements get NEG_INF."""
        b, lq = query_ids.shape
        pos = torch.arange(lq, device=query_ids.device)
        q = self.tok_emb(query_ids) + self.pos_emb(pos) + self.type_emb.weight[0]
        el = (
            self._masked_mean(el_text_ids) + self.ctx_proj(self._masked_mean(el_ctx_ids))
            + self.role_emb(el_role) + self.feat_mlp(el_feats) + self.type_emb.weight[1]
        )
        null = self.null.expand(b, 1, -1)
        x = torch.cat([q, null, el], dim=1)

        keep = torch.cat([query_ids != PAD_ID, el_mask.new_ones(b, 1), el_mask], dim=1)
        key_bias = (~keep).to(x.dtype)[:, None, None, :] * _MASK_BIAS
        for block in self.blocks:
            x = block(x, key_bias)
        logits = self.head(self.final_norm(x[:, lq:])).squeeze(-1).float()
        return logits.masked_fill(~torch.cat([el_mask.new_ones(b, 1), el_mask], dim=1), NEG_INF)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def listwise_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    """Mean of -log(sum of softmax probability over positives); index 0 is NULL."""
    pos = logits.masked_fill(~positive_mask, float("-inf")).logsumexp(dim=-1)
    return (logits.logsumexp(dim=-1) - pos).mean()


def save_checkpoint(model: Grounder, path: str | Path, tokenizer_path: str | Path) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(model.cfg), "tokenizer": str(tokenizer_path)},
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Grounder, Tokenizer, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Grounder(GrounderConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), load_tokenizer(ckpt["tokenizer"]), ckpt
