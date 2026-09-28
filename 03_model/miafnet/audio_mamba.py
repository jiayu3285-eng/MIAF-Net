"""Minimal Audio Mamba encoder used by the MIAF-Net audio branch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class MinimalMambaBlock(nn.Module):
    """A compact Mamba-like temporal block for log-Mel acoustic sequences."""

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        u, gate = xz.chunk(2, dim=-1)

        u = u.transpose(1, 2)
        kernel = self.conv.kernel_size[0]
        u = F.pad(u, (kernel - 1, 0))
        u = self.conv(u).transpose(1, 2)
        u = F.silu(u)

        y = self._selective_scan(u)
        y = y * F.silu(gate)
        y = self.out_proj(y)
        y = self.drop(y)
        return residual + y

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        batch, steps, dim_inner = u.shape
        a = -torch.exp(self.A_log.float())
        d = self.D.float()

        projected = self.x_proj(u)
        delta, b_t, c_t = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        state = u.new_zeros((batch, dim_inner, self.d_state))
        outputs = []
        for step in range(steps):
            delta_step = delta[:, step, :]
            u_step = u[:, step, :]
            b_step = b_t[:, step, :]
            c_step = c_t[:, step, :]

            delta_a = torch.exp(delta_step.unsqueeze(-1) * a.unsqueeze(0))
            delta_b_u = (delta_step * u_step).unsqueeze(-1) * b_step.unsqueeze(1)
            state = delta_a * state + delta_b_u

            y_step = (state * c_step.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_step)

        y = torch.stack(outputs, dim=1)
        return y + u * d


class AudioMambaEncoder(nn.Module):
    """Encode log-Mel spectrograms into a pooled acoustic representation."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(feature_dim, hidden_size)
        dt_rank = max(4, hidden_size // 32)
        self.blocks = nn.ModuleList(
            [
                MinimalMambaBlock(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x.mean(dim=1)
