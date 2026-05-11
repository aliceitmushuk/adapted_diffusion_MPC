"""
Clean multidimensional sequential diffusion module.

This file is a simplified replacement for the sequential part of
`diffusion_factor_model.py`. It removes the image U-Net / GaussianDiffusion
code and keeps only:

    - ConditionalTransformer
    - SequentialGaussianDiffusion
    - WarmUpCosineAnnealingWarmRestarts
    - Trainer

The model supports vector-valued sequential data:

    sequences.shape == [batch, seq_len, feature_dim]

Sampling is sequential over time indices, but each generated time step is a
vector in R^{feature_dim}.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from accelerate import Accelerator
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install accelerate: pip install accelerate") from exc

try:
    from ema_pytorch import EMA
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install ema-pytorch: pip install ema-pytorch") from exc

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install tqdm: pip install tqdm") from exc


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def exists(x: Any) -> bool:
    return x is not None


def default(val: Any, d: Union[Any, Callable[[], Any]]) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return t


def normalize_to_neg_one_to_one(x: Tensor) -> Tensor:
    return x * 2.0 - 1.0


def unnormalize_to_zero_to_one(t: Tensor) -> Tensor:
    return (t + 1.0) * 0.5


def extract(a: Tensor, t: Tensor, x_shape: Sequence[int]) -> Tensor:
    """Extract a[t] and reshape for broadcasting to x_shape."""
    batch = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch, *((1,) * (len(x_shape) - 1)))


# -----------------------------------------------------------------------------
# Positional / diffusion timestep embeddings
# -----------------------------------------------------------------------------


class SinusoidalPosEmb(Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


# -----------------------------------------------------------------------------
# Beta schedules
# -----------------------------------------------------------------------------


def linear_beta_schedule(timesteps: int) -> Tensor:
    scale = 1000.0 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine schedule from Nichol & Dhariwal."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


def sigmoid_beta_schedule(timesteps: int, start: float = -3, end: float = 3, tau: float = 1, clamp_min: float = 1e-5) -> Tensor:
    """Sigmoid schedule, useful for high-resolution diffusion settings."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


# -----------------------------------------------------------------------------
# Conditional Transformer for vector-valued sequential data
# -----------------------------------------------------------------------------


class ConditionalTransformer(Module):
    """
    Transformer denoiser for vector-valued sequential data.

    Input shape:
        values: [batch, seq_len, feature_dim]

    It predicts the diffusion target for one selected time index per batch:
        output: [batch, feature_dim]

    The causal mask prevents each time token from attending to future time tokens.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        feature_dim: int,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_bos_token: bool = False,
        use_alibi: bool = False,
        alibi_slope: float = 1.0,
        first_token_bias: float = 0.0,
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.dim = int(dim)
        self.use_bos_token = bool(use_bos_token)
        self.use_alibi = bool(use_alibi)
        self.alibi_slope = float(alibi_slope)
        self.first_token_bias = float(first_token_bias)

        self.value_proj = nn.Linear(self.feature_dim, dim)
        self.indicator_embed = nn.Embedding(2, dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        emb_len = self.seq_len + (1 if self.use_bos_token else 0)
        self.pos_emb = nn.Parameter(torch.randn(emb_len, dim) * 0.02)

        if self.use_bos_token:
            self.bos_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ff_mult),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

        mask = torch.triu(torch.ones(emb_len, emb_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        self.register_buffer("causal_mask", mask, persistent=False)

        if self.use_alibi or self.first_token_bias != 0.0:
            positions = torch.arange(emb_len)
            distances = positions.unsqueeze(0) - positions.unsqueeze(1)
            distances = distances.clamp_min(0).float()

            alibi_bias = torch.zeros(emb_len, emb_len)
            if self.use_alibi:
                alibi_bias -= self.alibi_slope * distances
            if self.first_token_bias != 0.0:
                alibi_bias[:, 0] += self.first_token_bias

            self.register_buffer("alibi_bias", alibi_bias, persistent=False)

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, self.feature_dim),
        )

    def forward(
        self,
        values: Tensor,
        target_indices: Tensor,
        timesteps: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            values: [batch, seq_len, feature_dim]
            target_indices: [batch]
            timesteps: [batch]
            key_padding_mask: [batch, seq_len], True means ignored.

        Returns:
            [batch, feature_dim]
        """
        if values.dim() != 3:
            raise ValueError("values must have shape [batch, seq_len, feature_dim]")

        batch, seq_len, feature_dim = values.shape
        device = values.device

        if seq_len != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {seq_len}")
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {feature_dim}")
        if target_indices.shape != (batch,):
            raise ValueError("target_indices must have shape [batch]")
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape [batch]")
        if key_padding_mask.shape != (batch, seq_len):
            raise ValueError("key_padding_mask must have shape [batch, seq_len]")

        tokens = self.value_proj(values)
        offset = 1 if self.use_bos_token else 0

        if self.use_bos_token:
            bos = self.bos_token.expand(batch, -1, -1)
            tokens = torch.cat([bos, tokens], dim=1)

        seq_len_tokens = tokens.shape[1]
        tokens = tokens + self.pos_emb[:seq_len_tokens].unsqueeze(0)

        indicator = torch.zeros(batch, seq_len_tokens, dtype=torch.long, device=device)
        indicator.scatter_(1, (target_indices + offset).unsqueeze(1), 1)
        tokens = tokens + self.indicator_embed(indicator)

        time_emb = self.time_mlp(timesteps.float())
        tokens[torch.arange(batch, device=device), target_indices + offset] += time_emb

        tokens = self.dropout(tokens)

        mask = self.causal_mask[:seq_len_tokens, :seq_len_tokens]
        if self.use_alibi or self.first_token_bias != 0.0:
            mask = mask + self.alibi_bias[:seq_len_tokens, :seq_len_tokens]

        if self.use_bos_token:
            key_padding_mask = F.pad(key_padding_mask, (1, 0), value=False)

        encoded = self.encoder(tokens, mask=mask, src_key_padding_mask=key_padding_mask)
        target_states = encoded[torch.arange(batch, device=device), target_indices + offset]
        return self.output(target_states)


# -----------------------------------------------------------------------------
# Sequential vector-valued Gaussian diffusion
# -----------------------------------------------------------------------------


class SequentialGaussianDiffusion(Module):
    """
    Sequential diffusion model for vector-valued time series.

    Training input:
        sequences: [batch, seq_len, feature_dim]

    Sampling output:
        samples: [batch_size, seq_len, feature_dim]

    The outer sampler generates time steps sequentially:
        X_1, X_2, ..., X_T

    but each generated X_t is vector-valued:
        X_t in R^{feature_dim}.
    """

    def __init__(
        self,
        model: Module,
        *,
        seq_len: int,
        feature_dim: int,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        ddim_eta: float = 0.0,
        objective: str = "pred_noise",
        beta_schedule: str = "cosine",
        schedule_fn_kwargs: Optional[dict] = None,
        auto_normalize: bool = False,
    ):
        super().__init__()

        self.model = model
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.objective = objective

        schedule_fn_kwargs = default(schedule_fn_kwargs, dict)

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule!r}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(betas.shape[0])

        def register_buffer(name: str, val: Tensor) -> None:
            self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        if objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1.0)
        else:
            raise ValueError(f"unknown objective {objective!r}")
        register_buffer("loss_weight", loss_weight)

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.sampling_timesteps = self.num_timesteps if sampling_timesteps is None else int(sampling_timesteps)
        if self.sampling_timesteps <= 0:
            raise ValueError("sampling_timesteps must be a positive integer")

        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = float(ddim_eta)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def build_context(self, sequences: Tensor, target_indices: Tensor, target_values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Build causal context.

        Args:
            sequences: [batch, seq_len, feature_dim]
            target_indices: [batch]
            target_values: [batch, feature_dim]

        Returns:
            context: [batch, seq_len, feature_dim]
            key_padding_mask: [batch, seq_len], True means ignored.
        """
        if sequences.dim() != 3:
            raise ValueError("sequences must have shape [batch, seq_len, feature_dim]")

        device = sequences.device
        batch, seq_len, feature_dim = sequences.shape

        if target_indices.shape != (batch,):
            raise ValueError("target_indices must have shape [batch]")
        if target_values.shape != (batch, feature_dim):
            raise ValueError("target_values must have shape [batch, feature_dim]")

        arange = torch.arange(seq_len, device=device)

        prefix_mask = arange.unsqueeze(0) < target_indices.unsqueeze(1)
        prefix_mask = prefix_mask.unsqueeze(-1)

        context = torch.zeros_like(sequences)
        context = torch.where(prefix_mask, sequences, context)
        context[torch.arange(batch, device=device), target_indices, :] = target_values

        key_padding_mask = arange.unsqueeze(0) > target_indices.unsqueeze(1)
        return context, key_padding_mask

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_v(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_v(self, x_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_noise_from_v(self, x_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, sequences: Tensor, t: Tensor, target_indices: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        batch = sequences.shape[0]
        device = sequences.device

        target = sequences[torch.arange(batch, device=device), target_indices, :]
        noise = default(noise, lambda: torch.randn_like(target))
        noisy_target = self.q_sample(target, t, noise)

        context, key_padding_mask = self.build_context(sequences, target_indices, noisy_target)
        pred = self.model(context, target_indices, t, key_padding_mask)

        if self.objective == "pred_noise":
            target_value = noise
        elif self.objective == "pred_x0":
            target_value = target
        elif self.objective == "pred_v":
            target_value = self.predict_v(target, t, noise)
        else:
            raise ValueError(f"unknown objective {self.objective!r}")

        loss = F.mse_loss(pred, target_value, reduction="none")
        loss = loss.mean(dim=-1)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, sequences: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        if sequences.dim() != 3:
            raise ValueError("sequences must have shape [batch, seq_len, feature_dim]")

        batch, seq_len, feature_dim = sequences.shape
        if seq_len != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {seq_len}")
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {feature_dim}")

        sequences = self.normalize(sequences)

        t = torch.randint(0, self.num_timesteps, (batch,), device=sequences.device).long()
        target_indices = torch.randint(0, self.seq_len, (batch,), device=sequences.device).long()
        return self.p_losses(sequences, t, target_indices, *args, **kwargs)

    def _build_ddim_time_pairs(self) -> Sequence[Tuple[int, int]]:
        times = torch.linspace(0, self.num_timesteps - 1, steps=self.sampling_timesteps, device=self.device)
        times = torch.flip(times.long(), dims=[0]).tolist()
        if len(times) == 1:
            return [(times[0], -1)]
        return list(zip(times[:-1], times[1:] + [-1]))

    def _model_predictions(
        self,
        context: Tensor,
        key_padding_mask: Tensor,
        target_indices: Tensor,
        x_t: Tensor,
        times: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        model_out = self.model(context, target_indices, times, key_padding_mask)

        if self.objective == "pred_noise":
            pred_noise = model_out
            x_start = self.predict_start_from_noise(x_t, times, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_out
            pred_noise = self.predict_noise_from_start(x_t, times, x_start)
        elif self.objective == "pred_v":
            x_start = self.predict_start_from_v(x_t, times, model_out)
            pred_noise = self.predict_noise_from_v(x_t, times, model_out)
        else:
            raise ValueError(f"unknown objective {self.objective!r}")

        return pred_noise, x_start

    def _ddpm_step(self, sequences: Tensor, pos: int) -> Tensor:
        batch_size = sequences.size(0)
        x_t = torch.randn(batch_size, self.feature_dim, device=self.device)
        target_indices = torch.full((batch_size,), pos, device=self.device, dtype=torch.long)

        for t in reversed(range(self.num_timesteps)):
            times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            context, key_padding_mask = self.build_context(sequences, target_indices, x_t)
            pred_noise, x_start = self._model_predictions(context, key_padding_mask, target_indices, x_t, times)

            if t == 0:
                x_t = x_start
            else:
                model_mean, _, log_variance = self.q_posterior(x_start, x_t, times)
                noise = torch.randn_like(x_t)
                x_t = model_mean + (0.5 * log_variance).exp() * noise

        return x_t

    def _ddim_step(self, sequences: Tensor, pos: int) -> Tensor:
        batch_size = sequences.size(0)
        x_t = torch.randn(batch_size, self.feature_dim, device=self.device)
        target_indices = torch.full((batch_size,), pos, device=self.device, dtype=torch.long)

        for time, time_next in self._build_ddim_time_pairs():
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            context, key_padding_mask = self.build_context(sequences, target_indices, x_t)
            pred_noise, x_start = self._model_predictions(context, key_padding_mask, target_indices, x_t, time_cond)

            if time_next < 0:
                x_t = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * (
                (1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)
            ).sqrt()
            c = (1.0 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x_t

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int = 16,
        save_timesteps: Optional[Sequence[int]] = None,
        return_all_timesteps: bool = False,
        conditioning: Optional[Tensor] = None,
        conditioning_mask: Optional[Tensor] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> Tensor:
        """
        Generate samples.

        Args:
            batch_size: number of generated sequences.
            conditioning: optional fixed values, shape [batch_size, seq_len, feature_dim]
                or [seq_len, feature_dim].
            conditioning_mask: bool mask, shape [batch_size, seq_len] or [seq_len].
                True means that time step is fixed and not generated.
            start_idx/end_idx: generate only within [start_idx, end_idx).

        Returns:
            [batch_size, seq_len, feature_dim]
        """
        if save_timesteps is not None:
            warnings.warn("`save_timesteps` is ignored by SequentialGaussianDiffusion.sample.", stacklevel=2)
        if return_all_timesteps:
            warnings.warn("`return_all_timesteps` is ignored; only final samples are returned.", stacklevel=2)

        sequences = torch.zeros(batch_size, self.seq_len, self.feature_dim, device=self.device)

        if conditioning is not None:
            conditioning = conditioning.to(self.device)
            if conditioning.dim() == 2:
                conditioning = conditioning.unsqueeze(0).expand(batch_size, -1, -1)
            if conditioning.shape != sequences.shape:
                raise ValueError("conditioning must have shape [batch_size, seq_len, feature_dim] or [seq_len, feature_dim]")
            if conditioning_mask is None:
                raise ValueError("conditioning_mask is required when conditioning is provided")

            conditioning_mask = conditioning_mask.to(self.device)
            if conditioning_mask.dim() == 1:
                conditioning_mask = conditioning_mask.unsqueeze(0).expand(batch_size, -1)
            if conditioning_mask.shape != (batch_size, self.seq_len):
                raise ValueError("conditioning_mask must have shape [batch_size, seq_len] or [seq_len]")

            conditioning_mask = conditioning_mask.bool()
            if not torch.equal(conditioning_mask, conditioning_mask[:1].expand_as(conditioning_mask)):
                raise ValueError("conditioning_mask must be identical across the batch for this sequential sampler")

            sequences = torch.where(conditioning_mask.unsqueeze(-1), conditioning, sequences)

        elif conditioning_mask is not None:
            raise ValueError("conditioning_mask was provided without conditioning")

        if end_idx is None:
            end_idx = self.seq_len
        start_idx = max(0, int(start_idx))
        end_idx = min(self.seq_len, int(end_idx))
        if start_idx >= end_idx:
            raise ValueError("Sampling window must include at least one index")

        if conditioning_mask is not None:
            conditioned_positions = conditioning_mask[0]
            indices_to_generate = [pos for pos in range(start_idx, end_idx) if not conditioned_positions[pos].item()]
        else:
            indices_to_generate = list(range(start_idx, end_idx))

        sample_pos = self._ddim_step if self.is_ddim_sampling else self._ddpm_step
        iterator: Iterable[int] = indices_to_generate
        if show_progress:
            iterator = tqdm(indices_to_generate, desc=progress_desc or "Sampling", unit="index", leave=False)

        for pos in iterator:
            sequences[:, pos, :] = sample_pos(sequences, pos)
            if show_progress:
                iterator.set_postfix(index=pos)

        return self.unnormalize(sequences)


# -----------------------------------------------------------------------------
# Warmup cosine scheduler and Trainer
# -----------------------------------------------------------------------------


class WarmUpCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    """CosineAnnealingWarmRestarts with a linear warmup phase."""

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        warmup_steps: int = 0,
    ):
        self.warmup_steps = int(warmup_steps)
        super().__init__(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * float(self.last_epoch + 1) / max(1, self.warmup_steps) for base_lr in self.base_lrs]
        return super().get_lr()


class Trainer:
    """
    Minimal trainer for SequentialGaussianDiffusion.

    Expects the dataset to produce either:
        tensor: [seq_len, feature_dim]
    or:
        tuple/list whose first element is [seq_len, feature_dim]

    The DataLoader batch should therefore be:
        [batch, seq_len, feature_dim]
    """

    def __init__(
        self,
        diffusion_model: Module,
        dataset,
        *,
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: Tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 16,
        results_folder: Union[str, os.PathLike] = "./results",
        amp: bool = False,
        mixed_precision_type: str = "fp16",
        split_batches: bool = True,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
        pin_memory: bool = True,
        cosine_scheduler: bool = True,
        warm_up: bool = True,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
        log_with_tensorboard: bool = True,
        save_best_loss: bool = True,
        sample_with_progress: bool = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision=mixed_precision_type if amp else "no")
        self.model = diffusion_model
        self.dataset = dataset
        self.train_batch_size = int(train_batch_size)
        self.gradient_accumulate_every = int(gradient_accumulate_every)
        self.train_num_steps = int(train_num_steps)
        self.save_and_sample_every = int(save_and_sample_every)
        self.num_samples = int(num_samples)
        self.max_grad_norm = float(max_grad_norm)
        self.save_best_loss = bool(save_best_loss)
        self.sample_with_progress = bool(sample_with_progress)

        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(parents=True, exist_ok=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if cosine_scheduler:
            if warm_up:
                self.scheduler = WarmUpCosineAnnealingWarmRestarts(
                    self.opt,
                    T_0=max(1, train_num_steps - warmup_steps),
                    T_mult=1,
                    eta_min=min_lr,
                    warmup_steps=warmup_steps,
                )
            else:
                self.scheduler = CosineAnnealingLR(self.opt, T_max=train_num_steps, eta_min=min_lr)
        else:
            self.scheduler = LambdaLR(self.opt, lr_lambda=lambda step: 1.0)

        self.model, self.opt, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.opt, self.dataloader, self.scheduler
        )

        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.accelerator.unwrap_model(self.model),
                beta=ema_decay,
                update_every=ema_update_every,
            ).to(self.accelerator.device)
        else:
            self.ema = None

        self.step = 0
        self.best_loss = float("inf")
        self.writer = SummaryWriter(log_dir=str(self.results_folder / "logs")) if (log_with_tensorboard and self.accelerator.is_main_process) else None

    @staticmethod
    def _unpack_batch(data: Any) -> Tensor:
        if isinstance(data, (tuple, list)):
            data = data[0]
        if not torch.is_tensor(data):
            raise TypeError("Dataset batch must be a tensor or a tuple/list whose first element is a tensor")
        return data

    def save(self, milestone: Union[int, str]) -> None:
        if not self.accelerator.is_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "best_loss": self.best_loss,
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, self.results_folder / f"model-{milestone}.pt")

    def load(self, milestone: Union[int, str]) -> None:
        device = self.accelerator.device
        data = torch.load(self.results_folder / f"model-{milestone}.pt", map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])
        self.step = int(data["step"])
        self.opt.load_state_dict(data["opt"])
        self.scheduler.load_state_dict(data["scheduler"])
        self.best_loss = float(data.get("best_loss", float("inf")))

        if self.ema is not None and data.get("ema") is not None:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data.get("scaler")):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    @torch.inference_mode()
    def save_and_sample(self, milestone: Union[int, str]) -> None:
        if not self.accelerator.is_main_process or self.ema is None:
            return

        self.ema.ema_model.eval()
        samples = self.ema.ema_model.sample(
            self.num_samples,
            show_progress=self.sample_with_progress,
            progress_desc=f"sampling-{milestone}",
        )
        torch.save(samples.detach().cpu(), self.results_folder / f"sample-{milestone}.pt")
        self.save(milestone)

    def train(self) -> None:
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    try:
                        data = next(self._data_iter)
                    except AttributeError:
                        self._data_iter = iter(self.dataloader)
                        data = next(self._data_iter)
                    except StopIteration:
                        self._data_iter = iter(self.dataloader)
                        data = next(self._data_iter)

                    data = self._unpack_batch(data).to(device)

                    with accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += float(loss.detach().item())

                    accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    if self.ema is not None:
                        self.ema.update()

                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", total_loss, self.step)
                        self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], self.step)

                    if total_loss < self.best_loss:
                        self.best_loss = total_loss
                        if self.save_best_loss:
                            self.save("best")

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save_and_sample(milestone)

                    pbar.set_description(f"loss: {total_loss:.4f}")

                self.step += 1
                pbar.update(1)

        if self.writer is not None:
            self.writer.close()

        accelerator.print("training complete")


__all__ = [
    "ConditionalTransformer",
    "SequentialGaussianDiffusion",
    "Trainer",
    "WarmUpCosineAnnealingWarmRestarts",
    "SinusoidalPosEmb",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "sigmoid_beta_schedule",
]
