import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MHA(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # self-attention
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class TemporalPatchify(nn.Module):
    """
    Splits (B, T, C) into patches along time:
    - patch_size: number of time steps per patch
    - stride: step between starts (can be < patch_size for overlap)
    Output shape: (B, N_patches, C * patch_size)
    """
    def __init__(self, patch_size: int, stride: Optional[int] = None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        ps, st = self.patch_size, self.stride
        if T < ps:
            # pad on the right
            pad_len = ps - T
            x = F.pad(x, (0, 0, 0, pad_len))  # pad time dimension
            T = ps
        # unfold: (B, C, T) -> patches
        x_ = x.transpose(1, 2)  # (B, C, T)
        patches = x_.unfold(dimension=2, size=ps, step=st)  # (B, C, N, ps)
        B, C, N, P = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, N, C * P)  # (B, N, C*ps)
        return patches


@dataclass
class TemporalViTConfig:
    in_channels: int                   # number of feature channels
    patch_size: int                    # time steps per patch
    stride: Optional[int] = None       # step between patch starts (None => no overlap)
    embed_dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    pos_dropout: float = 0.0
    use_cls_token: bool = True

    # Heads
    # TODO: confirm if we still keep other tasks
    task: Literal["classification", "regression", "forecast"] = "forecast"
    num_classes: int = 2               # for classification
    horizon: int = 24                  # for forecasting (time steps)
    forecast_mode: Literal["target_only", "multi"] = "target_only"
    target_dim: int = 0                # which channel to forecast if target_only
    out_channels: Optional[int] = None # set if forecast_mode == "multi"

class TemporalViT(nn.Module):
    def __init__(self, cfg: TemporalViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patchify = TemporalPatchify(cfg.patch_size, cfg.stride)
        patch_vec = cfg.in_channels * cfg.patch_size

        # projection + class token + positional embeddings
        self.proj = nn.Linear(patch_vec, cfg.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if cfg.use_cls_token else None
        self.pos_embed = None  # initialize lazily when seq length known
        self.pos_dropout = nn.Dropout(cfg.pos_dropout)

        # transformer encoder
        blocks = []
        for _ in range(cfg.depth):
            attn = PreNormResidual(cfg.embed_dim, MHA(cfg.embed_dim, heads=cfg.heads, dropout=cfg.dropout), dropout=cfg.dropout)
            ff = PreNormResidual(cfg.embed_dim, FeedForward(cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio), dropout=cfg.dropout), dropout=cfg.dropout)
            blocks += [attn, ff]
        self.encoder = nn.Sequential(*blocks)
        self.final_ln = nn.LayerNorm(cfg.embed_dim)

        # heads
        if cfg.task == "classification":
            self.head = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.embed_dim, cfg.num_classes),
            )
        elif cfg.task in ("regression", "forecast"):
            # forecast/regress head maps sequence tokens -> output
            # mean-pool tokens (or CLS) then MLP for regression
            out_dim = 1
            if cfg.task == "forecast":
                if cfg.forecast_mode == "target_only":
                    out_dim = cfg.horizon
                else:
                    assert cfg.out_channels is not None, "Set out_channels for multi-var forecasting."
                    out_dim = cfg.horizon * cfg.out_channels
            self.head = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.embed_dim, out_dim),
            )
        else:
            raise ValueError("Unknown task")

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02) if self.cls_token is not None else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _positional_embedding(self, n_tokens: int, device):
        # lazily create learnable position embeddings sized to current sequence length
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != n_tokens):
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, self.cfg.embed_dim, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        Returns:
          - classification: logits (B, num_classes)
          - regression: (B, 1)
          - forecast:
              target_only -> (B, horizon)
              multi -> (B, horizon, out_channels)
        """
        B, T, C = x.shape
        patches = self.patchify(x)                    # (B, N, C*ps)
        tokens = self.proj(patches)                  # (B, N, D)

        if self.cfg.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
            tokens = torch.cat([cls, tokens], dim=1) # (B, 1+N, D)

        pos = self._positional_embedding(tokens.shape[1], tokens.device)
        tokens = self.pos_dropout(tokens + pos)

        h = self.encoder(tokens)
        h = self.final_ln(h)

        if self.cfg.task == "classification":
            pooled = h[:, 0] if self.cfg.use_cls_token else h.mean(dim=1)
            return self.head(pooled)

        # regression / forecasting
        pooled = h[:, 0] if self.cfg.use_cls_token else h.mean(dim=1)
        out = self.head(pooled)

        if self.cfg.task == "forecast":
            if self.cfg.forecast_mode == "target_only":
                return out  # (B, horizon)
            else:
                B = out.shape[0]
                return out.view(B, self.cfg.horizon, self.cfg.out_channels)
        else:
            return out  # (B, 1)


# demo
def demo_classification():
    cfg = TemporalViTConfig(
        in_channels=16, patch_size=16, stride=16,
        embed_dim=256, depth=6, heads=8, mlp_ratio=4.0,
        task="classification", num_classes=5
    )
    model = TemporalViT(cfg)
    x = torch.randn(8, 256, 16)  # B, T, C
    logits = model(x)
    print("class logits:", logits.shape)  # (8, 5)


def demo_forecast_target_only():
    cfg = TemporalViTConfig(
        in_channels=32, patch_size=24, stride=12,  # overlapping patches
        embed_dim=256, depth=6, heads=8, mlp_ratio=4.0,
        task="forecast", horizon=24, forecast_mode="target_only", target_dim=0
    )
    model = TemporalViT(cfg)
    x = torch.randn(4, 240, 32)
    yhat = model(x)
    print("forecast (target):", yhat.shape)  # (4, 24)


def demo_forecast_multi():
    cfg = TemporalViTConfig(
        in_channels=32, patch_size=24, stride=12,
        embed_dim=256, depth=8, heads=8, mlp_ratio=4.0,
        task="forecast", horizon=24, forecast_mode="multi", out_channels=32
    )
    model = TemporalViT(cfg)
    x = torch.randn(2, 240, 32)
    yhat = model(x)
    print("forecast (multi):", yhat.shape)  # (2, 24, 32)


if __name__ == "__main__":
    demo_classification()
    demo_forecast_target_only()
    demo_forecast_multi()