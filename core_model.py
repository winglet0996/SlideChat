import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
from transformers import InstructBlipQFormerConfig
from transformers.models.instructblip.modeling_instructblip import InstructBlipQFormerEncoder
from typing import Optional, Tuple, Iterable, Literal, Dict, Sequence
ROUTE_FAMILIES = (
    'morphology_clinicopathology',
    'molecular_biomarker',
    'protein_program',
    'transcriptomic_program',
    'immune_microenvironment',
    'outcome',
)


class RoutedLoRAAdapter(nn.Module):
    """A small additive LoRA branch kept in FP32 by default.

    The up projection is zero-initialized so adding routed modules preserves
    the pre-routed model output at initialization.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'LoRA rank must be positive, got {rank}.')
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class RoutedLoRALinear(nn.Module):
    """Frozen linear + shared LoRA + per-sample family LoRA.

    ``route_family`` is installed by :class:`LLaVAModel_qwen3_5` for the
    duration of one forward/generation call. Routing always indexes dimension
    zero. For flattened ``(batch * tokens, hidden)`` inputs, routes are
    expanded with ``repeat_interleave``; no token-wise router is used.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float,
                 dropout: float, route_families: Sequence[str] = ROUTE_FAMILIES,
                 family_rank: Optional[int] = None,
                 family_alpha: Optional[float] = None):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f'Expected nn.Linear, got {type(base_linear)!r}.')
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)
        family_rank = rank if family_rank is None else int(family_rank)
        family_alpha = alpha if family_alpha is None else float(family_alpha)
        self.shared_lora = RoutedLoRAAdapter(
            base_linear.in_features, base_linear.out_features, rank, alpha, dropout)
        self.family_lora = nn.ModuleDict({
            family: RoutedLoRAAdapter(
                base_linear.in_features, base_linear.out_features,
                family_rank, family_alpha, dropout)
            for family in route_families
        })
        self.route_families = tuple(route_families)
        self._route_family = None

    def set_route_family(self, route_family: Optional[torch.Tensor]) -> None:
        self._route_family = route_family

    def _route_for_input(self, x: torch.Tensor) -> torch.Tensor:
        route = self._route_family
        if route is None:
            raise RuntimeError(
                'RoutedLoRALinear received no route_family. '
                'Every routed forward must provide a per-sample route.')
        route = route.to(device=x.device, dtype=torch.long).view(-1)
        if x.ndim < 2:
            raise ValueError(f'Routed LoRA expects at least 2D input, got {tuple(x.shape)}.')
        if x.size(0) == route.numel():
            return route
        if route.numel() > 0 and x.size(0) % route.numel() == 0:
            # Transformer kernels may flatten batch and token dimensions, and
            # generation expands the leading batch dimension for beam search.
            # Both layouts keep rows for one sample/beam contiguous.
            return route.repeat_interleave(x.size(0) // route.numel())
        raise ValueError(
            f'Cannot align per-sample route shape {tuple(route.shape)} with '
            f'linear input shape {tuple(x.shape)}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_linear(x) + self.shared_lora(x)
        route = self._route_for_input(x)
        residual = torch.zeros_like(out)
        for family_id, family in enumerate(self.route_families):
            idx = (route == family_id).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            family_out = self.family_lora[family](x.index_select(0, idx))
            residual = residual.index_add(0, idx, family_out)
        return out + residual


def replace_linear_with_routed_lora(
    module: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    route_families: Sequence[str] = ROUTE_FAMILIES,
    family_rank: Optional[int] = None,
    family_alpha: Optional[float] = None,
) -> int:
    """Replace matching leaf linear modules in-place and return the count."""
    target_modules = set(str(name) for name in target_modules)
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in target_modules:
            setattr(module, child_name, RoutedLoRALinear(
                child, rank, alpha, dropout, route_families=route_families,
                family_rank=family_rank, family_alpha=family_alpha))
            replaced += 1
        else:
            replaced += replace_linear_with_routed_lora(
                child, target_modules, rank, alpha, dropout,
                route_families=route_families, family_rank=family_rank,
                family_alpha=family_alpha)
    return replaced
class InputComposer(nn.Module):
    """Pad variable-length multimodal inputs for Qwen-style decoder models."""

    def forward(
        self,
        embeds_list,
        labels_list,
        attention_list,
        position_ids_list,
        padding_side: str = 'right',
        label_pad_value: int = -100,
    ):
        if not embeds_list:
            raise ValueError("InputComposer received an empty batch.")

        max_len = max(x.size(0) for x in embeds_list)
        batch_size = len(embeds_list)
        hidden_dim = embeds_list[0].size(-1)
        device = embeds_list[0].device
        dtype = embeds_list[0].dtype
        label_dtype = labels_list[0].dtype
        has_mrope = position_ids_list[0].ndim == 2

        inputs_embeds = torch.zeros((batch_size, max_len, hidden_dim), dtype=dtype, device=device)
        labels = torch.full((batch_size, max_len), label_pad_value, dtype=label_dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        if has_mrope:
            position_ids = torch.zeros(
                (position_ids_list[0].size(0), batch_size, max_len),
                dtype=torch.long,
                device=device,
            )
        else:
            position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for b_idx, (emb, lbl, attn, pids) in enumerate(
            zip(embeds_list, labels_list, attention_list, position_ids_list)
        ):
            cur_len = emb.size(0)
            seq_slice = slice(0, cur_len) if padding_side == 'right' else slice(max_len - cur_len, max_len)
            inputs_embeds[b_idx, seq_slice] = emb
            labels[b_idx, seq_slice] = lbl
            attention_mask[b_idx, seq_slice] = attn.bool()
            if has_mrope:
                position_ids[:, b_idx, seq_slice] = pids
            else:
                position_ids[b_idx, seq_slice] = pids

        return {
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }


class MRoPEPositionIDGenerator(nn.Module):
    """Generate Qwen3.5 four-row position ids: text, temporal, height, width."""

    num_position_rows = 4

    def __init__(self):
        super().__init__()

    def forward(
        self,
        sequential_ids: torch.Tensor,
        patch_start: int,
        token_positions: torch.Tensor,
        token_valid: torch.Tensor,
        vision_end_index: int,
        base_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = sequential_ids.numel()
        if base_position_ids is None:
            pos = sequential_ids.unsqueeze(0).expand(self.num_position_rows, -1).clone()
        else:
            pos = base_position_ids.clone()

        if token_positions.numel() == 0:
            return pos

        rows = token_positions[:, 0].long()
        cols = token_positions[:, 1].long()
        valid = token_valid.bool()
        patch_len = token_positions.size(0)
        patch_slice = slice(patch_start, patch_start + patch_len)

        temporal_val = sequential_ids[patch_start]
        fallback = sequential_ids[patch_slice]
        pos[1, patch_slice] = torch.where(valid, temporal_val.expand_as(rows), fallback)
        pos[2, patch_slice] = torch.where(valid, rows, fallback)
        pos[3, patch_slice] = torch.where(valid, cols, fallback)
        rows_max = torch.where(valid, rows, temporal_val).max()
        cols_max = torch.where(valid, cols, temporal_val).max()
        max_resume = torch.stack([temporal_val, rows_max, cols_max]).max() + 1

        if vision_end_index < seq_len:
            tail = torch.arange(
                seq_len - vision_end_index,
                device=sequential_ids.device,
                dtype=sequential_ids.dtype,
            ) + max_resume
            pos[1:, vision_end_index:] = tail.unsqueeze(0).expand(3, -1)

        return pos


class PatchDecoderBlock(nn.Module):
    """Pre-norm query decoder block: self-attn, patch cross-attn, FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        attn_kwargs = dict(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(**attn_kwargs)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(**attn_kwargs)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_key_padding_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normed = self.self_norm(x)
        ctx, _ = self.self_attn(query=normed, key=normed, value=normed, need_weights=False)
        x = x + self.dropout(ctx)

        normed = self.cross_norm(x)
        ctx, attn_heads = self.cross_attn(
            query=normed,
            key=patch_tokens,
            value=patch_tokens,
            key_padding_mask=patch_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.dropout(ctx)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, attn_heads


class PromptConditionedPatchResampler(nn.Module):
    """Single-stage prompt-conditioned decoder stack for WSI patch grids."""

    def __init__(
        self,
        patch_dim: int = 768,
        llm_hidden_size: int = 2560,
        resampler_dim: int = 1024,
        num_query: Optional[int] = None,
        num_layers: int = 2,
        num_region_tokens: Optional[int] = None,
        num_visual_tokens: Optional[int] = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_local_conv: bool = True,
        query_init_std: float = 0.5,
        route_families: Sequence[str] = ROUTE_FAMILIES,
    ):
        super().__init__()
        if resampler_dim % num_heads != 0:
            raise ValueError(f"resampler_dim={resampler_dim} must be divisible by num_heads={num_heads}.")

        self.patch_dim = int(patch_dim)
        self.llm_hidden_size = int(llm_hidden_size)
        self.resampler_dim = int(resampler_dim)
        if num_query is None:
            num_query = num_visual_tokens if num_visual_tokens is not None else num_region_tokens
        self.num_query = 8 if num_query is None else int(num_query)
        self.num_layers = int(num_layers)
        self.query_init_std = float(query_init_std)
        self.route_families = tuple(route_families)
        if not self.route_families:
            raise ValueError('route_families must not be empty.')
        if self.num_query <= 0:
            raise ValueError(f"num_query must be positive, got {self.num_query}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        if self.query_init_std <= 0:
            raise ValueError(f"query_init_std must be positive, got {self.query_init_std}.")

        self.patch_proj = nn.Linear(self.patch_dim, self.resampler_dim)
        self.patch_norm = nn.LayerNorm(self.resampler_dim)
        self.coord_proj = nn.Sequential(
            nn.Linear(4, self.resampler_dim),
            nn.GELU(),
            nn.Linear(self.resampler_dim, self.resampler_dim),
        )
        if use_local_conv:
            self.local_mixer = nn.Sequential(
                nn.Conv2d(self.resampler_dim, self.resampler_dim, 3, padding=1,
                          groups=self.resampler_dim, bias=False),
                nn.GELU(),
                nn.Conv2d(self.resampler_dim, self.resampler_dim, 1, bias=True),
            )
        else:
            self.local_mixer = None

        self.prompt_proj = nn.Linear(self.llm_hidden_size, self.resampler_dim)
        self.prompt_norm = nn.LayerNorm(self.resampler_dim)
        self.query_tokens = nn.Parameter(torch.empty(self.num_query, self.resampler_dim))
        self.query_id = nn.Parameter(torch.empty(self.num_query, self.resampler_dim))
        self.family_query_residual = nn.ParameterDict({
            family: nn.Parameter(torch.zeros(self.num_query, self.resampler_dim))
            for family in self.route_families
        })
        attn_kwargs = dict(embed_dim=self.resampler_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.prompt_attn = nn.MultiheadAttention(**attn_kwargs)
        self.decoder_blocks = nn.ModuleList([
            PatchDecoderBlock(self.resampler_dim, num_heads, dropout)
            for _ in range(self.num_layers)
        ])
        self.to_llm = nn.Linear(self.resampler_dim, self.llm_hidden_size)
        self.output_norm = nn.LayerNorm(self.llm_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.query_tokens, std=self.query_init_std)
        trunc_normal_(self.query_id, std=self.query_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_output_rms(self, rms: float) -> None:
        if rms is None or not math.isfinite(float(rms)) or float(rms) <= 0:
            return
        with torch.no_grad():
            nn.init.constant_(self.output_norm.weight, float(rms))
            nn.init.zeros_(self.output_norm.bias)

    @staticmethod
    def _shape_valid_mask(features: torch.Tensor, feature_shapes: Optional[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = features.shape
        device = features.device
        if feature_shapes is None:
            shape_mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        else:
            feature_shapes = feature_shapes.to(device=device)
            rows = torch.arange(h, device=device).view(1, h, 1)
            cols = torch.arange(w, device=device).view(1, 1, w)
            valid_h = feature_shapes[:, 0].view(b, 1, 1).clamp(min=1, max=h)
            valid_w = feature_shapes[:, 1].view(b, 1, 1).clamp(min=1, max=w)
            shape_mask = (rows < valid_h) & (cols < valid_w)
        nonzero_mask = features.detach().float().abs().sum(dim=1) > 0
        valid_mask = shape_mask & nonzero_mask
        empty = ~valid_mask.flatten(1).any(dim=1)
        if empty.any():
            valid_mask[empty, 0, 0] = True
        return valid_mask

    @staticmethod
    def _coord_features(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(height, device=device, dtype=torch.float32)
        cols = torch.arange(width, device=device, dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        norm_r = grid_r / max(height - 1, 1)
        norm_c = grid_c / max(width - 1, 1)
        coords = torch.stack([norm_r, norm_c, norm_r * 2 - 1, norm_c * 2 - 1], dim=-1)
        return coords.view(1, height * width, 4).expand(batch_size, -1, -1)

    @staticmethod
    def _coord_indices(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(height, device=device)
        cols = torch.arange(width, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        coords = torch.stack([grid_r, grid_c], dim=-1)
        return coords.view(1, height * width, 2).expand(batch_size, -1, -1)

    @staticmethod
    def _safe_prompt_mask(prompt_embeds: torch.Tensor, prompt_attention_mask: Optional[torch.Tensor]):
        b, l, _ = prompt_embeds.shape
        if prompt_attention_mask is None:
            mask = torch.ones((b, l), dtype=torch.bool, device=prompt_embeds.device)
        else:
            mask = prompt_attention_mask.to(device=prompt_embeds.device).bool()
        empty = ~mask.any(dim=1)
        if empty.any():
            prompt_embeds = prompt_embeds.clone()
            prompt_embeds[empty, 0] = 0
            mask = mask.clone()
            mask[empty, 0] = True
        return prompt_embeds, mask

    def _normalize_route_family(self, route_family, batch_size: int, device: torch.device):
        if route_family is None:
            raise ValueError(
                'Missing route_family. PromptConditionedPatchResampler requires '
                'one route id per image.')
        if torch.is_tensor(route_family):
            route_ids = route_family.to(device=device, dtype=torch.long).view(-1)
        else:
            route_ids = torch.as_tensor(route_family, device=device, dtype=torch.long).view(-1)
        if route_ids.numel() != batch_size:
            raise ValueError(
                f'route_family must be per-image with shape [{batch_size}], '
                f'got {tuple(route_ids.shape)}.')
        if bool(((route_ids < 0) | (route_ids >= len(self.route_families))).any().item()):
            raise ValueError(f'Invalid route_family ids: {route_ids.tolist()}')
        return route_ids

    def _condition_queries(self, prompt_tokens, prompt_mask, route_family=None):
        batch_size = prompt_tokens.size(0)
        route_ids = self._normalize_route_family(
            route_family, batch_size, prompt_tokens.device)
        shared_queries = (self.query_tokens + self.query_id).unsqueeze(0).expand(
            batch_size, -1, -1)
        residual = torch.zeros_like(shared_queries)
        for family_id, family in enumerate(self.route_families):
            idx = (route_ids == family_id).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            family_residual = self.family_query_residual[family].unsqueeze(0).expand(
                idx.numel(), -1, -1)
            residual = residual.index_add(0, idx, family_residual)
        queries = shared_queries + residual
        conditioned, _ = self.prompt_attn(
            query=queries,
            key=prompt_tokens,
            value=prompt_tokens,
            key_padding_mask=~prompt_mask,
            need_weights=False,
        )
        return queries + self.dropout(conditioned)

    @staticmethod
    def _raise_if_nonfinite(name: str, value: torch.Tensor, extra: str = "") -> None:
        if torch.isfinite(value).all():
            return
        value32 = value.detach().float()
        finite = value32[torch.isfinite(value32)]
        if finite.numel() > 0:
            min_value = float(finite.min().item())
            max_value = float(finite.max().item())
        else:
            min_value = float('nan')
            max_value = float('nan')
        suffix = f" {extra}" if extra else ""
        raise RuntimeError(
            f"Non-finite tensor detected in {name}:{suffix} "
            f"shape={tuple(value.shape)} dtype={value.dtype} "
            f"min={min_value:.6g} max={max_value:.6g}"
        )

    def forward(
        self,
        features: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        feature_shapes: Optional[torch.Tensor] = None,
        route_family: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(f"Expected features with shape (B, C, H, W), got {tuple(features.shape)}.")
        if features.size(1) != self.patch_dim:
            raise ValueError(f"Expected patch_dim={self.patch_dim}, got C={features.size(1)}.")

        b, _, h, w = features.shape
        prompt_embeds, prompt_attention_mask = self._safe_prompt_mask(prompt_embeds, prompt_attention_mask)
        valid_mask = self._shape_valid_mask(features, feature_shapes)

        patch_tokens = features.permute(0, 2, 3, 1).reshape(b, h * w, self.patch_dim)
        patch_tokens = self.patch_norm(self.patch_proj(patch_tokens))
        coords = self._coord_features(b, h, w, features.device).to(dtype=patch_tokens.dtype)
        patch_tokens = patch_tokens + self.coord_proj(coords)

        if self.local_mixer is not None:
            grid_tokens = patch_tokens.view(b, h, w, self.resampler_dim).permute(0, 3, 1, 2).contiguous()
            grid_tokens = grid_tokens * valid_mask.unsqueeze(1).to(dtype=grid_tokens.dtype)
            mixed = self.local_mixer(grid_tokens)
            mixed = mixed * valid_mask.unsqueeze(1).to(dtype=mixed.dtype)
            patch_tokens = patch_tokens + mixed.permute(0, 2, 3, 1).reshape(b, h * w, self.resampler_dim)

        self._raise_if_nonfinite('prompt_embeds', prompt_embeds)
        prompt_dtype = self.prompt_proj.weight.dtype
        prompt_embeds_fp32 = prompt_embeds.float()
        prompt_proj_weight = self.prompt_proj.weight.float()
        prompt_proj_bias = self.prompt_proj.bias.float() if self.prompt_proj.bias is not None else None
        prompt_tokens = F.linear(prompt_embeds_fp32, prompt_proj_weight, prompt_proj_bias)
        prompt_tokens = F.layer_norm(
            prompt_tokens,
            self.prompt_norm.normalized_shape,
            self.prompt_norm.weight.float(),
            self.prompt_norm.bias.float(),
            self.prompt_norm.eps,
        )
        self._raise_if_nonfinite('prompt_tokens', prompt_tokens, extra='after_prompt_proj_norm')
        prompt_tokens = prompt_tokens.to(dtype=prompt_dtype)

        patch_key_padding = ~valid_mask.flatten(1)
        visual_tokens = self._condition_queries(
            prompt_tokens, prompt_attention_mask, route_family=route_family)
        patch_attn_heads = None
        for layer_idx, block in enumerate(self.decoder_blocks):
            visual_tokens, attn_heads = block(
                visual_tokens,
                patch_tokens,
                patch_key_padding,
                need_weights=layer_idx == len(self.decoder_blocks) - 1,
            )
            if attn_heads is not None:
                patch_attn_heads = attn_heads
        if patch_attn_heads is None:
            raise RuntimeError("Decoder stack did not return final cross-attention weights.")

        patch_attention_heads = patch_attn_heads.float()
        patch_attention_heads = patch_attention_heads.masked_fill(
            ~valid_mask.flatten(1).unsqueeze(1).unsqueeze(1), 0.0)
        patch_attention_heads = patch_attention_heads / patch_attention_heads.sum(
            dim=-1, keepdim=True).clamp_min(1e-6)

        patch_attention = patch_attention_heads.mean(dim=1).float()
        patch_attention = patch_attention.masked_fill(~valid_mask.flatten(1).unsqueeze(1), 0.0)
        patch_attention = patch_attention / patch_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        coord_indices = self._coord_indices(b, h, w, features.device).float()
        token_positions = torch.bmm(patch_attention, coord_indices).round().long()
        token_positions[..., 0].clamp_(0, h - 1)
        token_positions[..., 1].clamp_(0, w - 1)
        token_valid = torch.ones((b, self.num_query), dtype=torch.bool, device=features.device)
        llm_tokens = self.output_norm(self.to_llm(visual_tokens))

        return {
            'visual_tokens': llm_tokens,
            'token_positions': token_positions,
            'token_valid': token_valid,
            'patch_attention': patch_attention.view(b, self.num_query, h, w),
            'patch_valid_mask': valid_mask,
            'region_attention': patch_attention,
            'region_attention_heads': patch_attention_heads,
            'visual_to_region_attention': None,
            'visual_to_region_attention_heads': None,
        }
class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_mult: float = 0.0):
        super().__init__()
        if hidden_mult > 0:
            hidden_dim = int(in_dim * hidden_mult)
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.head = nn.Linear(in_dim, 1)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small gain for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller gain for the final regression output
                gain = 0.1 if m.out_features == 1 else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SurvivalHead(nn.Module):
    """
    Unified Survival Head supporting both Cox and Discrete-Time (bin-based) methods.
    
    Methods:
    - 'cox': Outputs a single risk score (log-hazard) per sample for Cox PH loss.
             Higher risk score = higher risk of event.
    - 'discrete': Outputs K hazard logits for discrete time intervals.
                  Uses logistic hazard loss and can predict survival curves.
    
    Args:
        in_dim: Input dimension (hidden state dimension)
        method: 'cox' for Cox PH or 'discrete' for discrete-time survival
        num_intervals: Number of time intervals (K) for discrete method (ignored for cox)
        time_intervals: Optional tensor of K+1 time boundaries for discrete method
    """

    def __init__(
        self,
        in_dim: int,
        method: Literal["cox", "discrete"] = "cox",
        num_intervals: int = 6,
        time_intervals: Optional[torch.Tensor] = None,
        hidden_mult: float = 0.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.method = method
        self.num_intervals = num_intervals
        out_dim = 1 if method == "cox" else num_intervals
        
        if hidden_mult > 0:
            hidden_dim = int(in_dim * hidden_mult)
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(in_dim, out_dim)
            ) if dropout > 0 else nn.Linear(in_dim, out_dim)
            
        # Register time intervals for discrete method (optional, only for median prediction)
        if time_intervals is not None:
            self.register_buffer("time_intervals", time_intervals.float())
        else:
            # No time intervals - median survival prediction will not be available
            self.register_buffer("time_intervals", None)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns raw logits without clamping to preserve gradients.
        Numerical stability is handled in the loss functions.
        
        For cox: returns risk scores (N,)
        For discrete: returns hazard logits (N, K)
        """
        logits = self.head(x)
        if self.method == "cox":
            return logits.squeeze(-1)  # (N,)
        else:
            return logits  # (N, K)

    # ========== Cox-specific methods ==========
    
    @torch.no_grad()
    def predict_risk_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return risk scores (N,), where larger means higher risk.
        
        For cox: directly uses the output.
        For discrete: uses cumulative hazard as risk (sum of interval hazards).
        """
        if self.method == "cox":
            return self.forward(x)
        else:
            # For discrete, use cumulative hazard as risk
            return self._compute_discrete_risk_scores(x, mode="cumhaz")
    
    # ========== Discrete-specific methods ==========
    
    def predict_hazards(self, x: torch.Tensor) -> torch.Tensor:
        """Return hazards h_k in (0,1). Shape: (N, K). Only for discrete method."""
        if self.method != "discrete":
            raise ValueError("predict_hazards only available for discrete method")
        return torch.sigmoid(self.forward(x))

    def _hazards_and_survival_end(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns for discrete method:
          hazards: (N, K)
          S_end:   (N, K) = [S(t1), ..., S(tK)]
        """
        hazards = torch.sigmoid(self.forward(x))         # (N, K)
        S_end = torch.cumprod(1.0 - hazards, dim=1)      # (N, K)
        return hazards, S_end

    @torch.no_grad()
    def predict_survival_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Survival at end of each interval: (N, K) = [S(t1), ..., S(tK)].
        Only for discrete method."""
        if self.method != "discrete":
            raise ValueError("predict_survival_probs only available for discrete method")
        _, S_end = self._hazards_and_survival_end(x)
        return S_end

    @torch.no_grad()
    def _compute_discrete_risk_scores(
        self,
        x: torch.Tensor,
        mode: Literal["cumhaz", "nll_final"] = "cumhaz",
    ) -> torch.Tensor:
        """
        Return risk scores (N,), where larger means higher risk.

        mode="cumhaz": Cumulative hazard = Σ_k h_k. Simple and doesn't need time_intervals.
        mode="nll_final": risk = -log S(t_K)
        """
        hazards, S_end = self._hazards_and_survival_end(x)

        if mode == "nll_final":
            final_survival = S_end[:, -1]
            return -torch.log(final_survival + EPS)

        if mode == "cumhaz":
            # Cumulative hazard: sum of hazards - higher = more risky
            # This doesn't require time_intervals
            cumhaz = hazards.sum(dim=1)  # (N,)
            return cumhaz

        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def predict_median_survival_time(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Predict median survival time per sample (discrete method only):
        returns the first boundary time t_k where S(t_k) < 0.5,
        otherwise returns the last boundary t_K.

        Shape: (N,) or None if time_intervals not available.
        """
        if self.method != "discrete":
            raise ValueError("predict_median_survival_time only available for discrete method")
        
        if self.time_intervals is None:
            # Cannot compute actual survival times without interval boundaries
            return None
        
        S_end = self.predict_survival_probs(x)          # (N, K)
        below = S_end < 0.5                             # (N, K)
        first_idx = torch.argmax(below.to(torch.int64), dim=1)  # (N,)

        never = ~below.any(dim=1)                       # (N,)
        first_idx = torch.where(
            never,
            torch.full_like(first_idx, self.num_intervals - 1),
            first_idx,
        )

        median_times = self.time_intervals[1:][first_idx]  # (N,)
        return median_times


# ========== Loss Functions ==========

def cox_ph_loss(
    theta: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Cox proportional hazards partial likelihood loss with numerical stability.
    
    Uses centered theta for numerical stability while preserving gradients.
    
    Args:
        theta: (N,) risk score (log-risk)
        time:  (N,) follow-up time
        event: (N,) 1=event, 0=censored
        eps:   Small constant for numerical stability
    """
    theta = theta.view(-1)
    time = time.view(-1)
    event = event.view(-1).float()
    
    N = theta.numel()
    n_events = event.sum()
    
    if n_events < 1:
        # No events: return small L2 regularization to keep gradients flowing
        # This prevents gradient vanishing while not affecting the model much
        return 1e-6 * (theta ** 2).mean()

    # Center theta for numerical stability (subtract max)
    # This is mathematically equivalent but prevents exp() overflow
    theta_max = theta.detach().max()  # detach to avoid affecting gradients
    theta_centered = theta - theta_max

    # Risk set mask: R[i,j] = (time[j] >= time[i])
    # Sample j is in risk set of sample i if j's time >= i's time
    R = (time[None, :] >= time[:, None])  # (N, N)

    # log denom_i = log sum_{j in R_i} exp(theta_j)
    # = log sum_{j in R_i} exp(theta_centered_j + theta_max)
    # = theta_max + log sum_{j in R_i} exp(theta_centered_j)
    theta_row = theta_centered.view(1, N).expand(N, N)
    
    # Mask out samples not in risk set with large negative value
    masked_theta = torch.where(R, theta_row, torch.tensor(-1e9, device=theta.device, dtype=theta.dtype))
    
    # logsumexp is numerically stable
    log_denom = torch.logsumexp(masked_theta, dim=1)  # (N,)
    # Add back theta_max: log_denom_true = theta_max + log_denom
    # But since we also centered theta in numerator, they cancel out:
    # theta - log_denom_true = (theta_centered + theta_max) - (theta_max + log_denom) = theta_centered - log_denom
    
    # Compute per-sample negative log partial likelihood
    # loss_i = -(theta_i - log_denom_i) for event samples
    per_sample_nll = -(theta_centered - log_denom)  # (N,)
    
    # Only sum over event samples
    loss = (per_sample_nll * event).sum() / (n_events + eps)
    
    # Safety check: if loss is NaN/Inf, return small regularization loss
    if not torch.isfinite(loss):
        return 1e-6 * (theta ** 2).mean()
    
    return loss


def logistic_hazard_loss(
    logits: torch.Tensor,
    target_y: torch.Tensor,
    at_risk_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Standard discrete-time logistic hazard loss.

    Args:
        logits:       (N, K) hazard logits from SurvivalHead (discrete mode)
        target_y:     (N, K) binary, y_{ik}=1 only at event interval (if event observed)
        at_risk_mask: (N, K) 1 where that interval contributes to likelihood
        
    Returns:
        Scalar loss value
    """
    bce = F.binary_cross_entropy_with_logits(logits, target_y, reduction="none")  # (N, K)
    return (bce * at_risk_mask).sum() / (at_risk_mask.sum() + EPS)


def prepare_discrete_survival_targets(
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
    time_intervals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare target_y and at_risk_mask for the discrete-time hazard model.

    Args:
        event_times: (N,) survival times
        event_indicators: (N,) 1=event, 0=censored
        time_intervals: (K+1,) boundaries [t0, t1, ..., tK]
        
    Returns:
        target_y: (N, K) binary target
        at_risk_mask: (N, K) mask for loss computation
    """
    if time_intervals.ndim != 1 or time_intervals.numel() < 2:
        raise ValueError("time_intervals must be 1D with length >= 2 (boundaries).")

    N = int(event_times.numel())
    K = int(time_intervals.numel() - 1)
    device = event_times.device

    target_y = torch.zeros(N, K, device=device)
    at_risk_mask = torch.zeros(N, K, device=device)

    for i in range(N):
        t_i = event_times[i]
        d_i = event_indicators[i].item()

        m = torch.searchsorted(time_intervals[1:], t_i, right=False)
        m = torch.clamp(m, 0, K - 1).item()

        if d_i == 1:  # event observed in interval m
            at_risk_mask[i, :m] = 1.0
            target_y[i, :m] = 0.0
            at_risk_mask[i, m] = 1.0
            target_y[i, m] = 1.0
        else:  # censored in interval m => contribute through m with y=0
            at_risk_mask[i, : m + 1] = 1.0
            target_y[i, : m + 1] = 0.0

    return target_y, at_risk_mask
class WSIProjectorMLP(nn.Module):
    """Single MLP projector for one WSI encoder source."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_mult: float = 2.0):
        """
        Initialize the MLP projector.
        
        Args:
            input_dim: Input dimension from WSI encoder
            output_dim: Output dimension (LLM hidden size)
            hidden_mult: Multiplier for hidden layer dimension. 
                         If <= 0, uses a single linear layer.
        """
        super().__init__()
        
        if hidden_mult > 0:
            hidden_dim = int(output_dim * hidden_mult)
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.layers = nn.Linear(input_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, input_dim) or (input_dim,)
            
        Returns:
            Projected tensor of shape (B, output_dim) or (output_dim,)
        """
        return self.layers(x)


class WSIProjector(nn.Module):
    """
    Multi-source WSI Feature Projector.
    
    Projects WSI-level features from multiple encoders (e.g., TITAN, CONCH, UNI)
    to the LLM hidden dimension using separate MLPs for each source.
    
    The projected features can be used as:
    1. Soft prompts: Inserted into the LLM input sequence before patch-level tokens
    2. DeepStack injection: Concatenated with patch features at intermediate layers
    
    Args:
        wsi_input_dims: List of input dimensions for each WSI encoder source.
                        E.g., [768, 1024, 768] for three different encoders.
        llm_hidden_size: LLM hidden dimension to project to.
        hidden_mult: Multiplier for hidden layer size in each MLP.
        dropout: Dropout rate applied after projection.
    """
    
    def __init__(
        self,
        wsi_input_dims: list,
        llm_hidden_size: int,
        hidden_mult: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not wsi_input_dims:
            raise ValueError("wsi_input_dims cannot be empty")
        
        self.wsi_input_dims = wsi_input_dims
        self.llm_hidden_size = llm_hidden_size
        self.num_sources = len(wsi_input_dims)
        
        # Create separate MLPs for each WSI source
        self.projectors = nn.ModuleList([
            WSIProjectorMLP(dim, llm_hidden_size, hidden_mult)
            for dim in wsi_input_dims
        ])

        # Match the scale of LLM token embeddings after projection.
        self.post_norms = nn.ModuleList([
            nn.LayerNorm(llm_hidden_size, elementwise_affine=True)
            for _ in wsi_input_dims
        ])
        for norm in self.post_norms:
            nn.init.constant_(norm.weight, 1.0 / math.sqrt(llm_hidden_size))
            nn.init.constant_(norm.bias, 0.0)
        
        # Optional dropout after projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        wsi_features: list,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Project multiple WSI features to LLM hidden dimension.
        
        Args:
            wsi_features: List of tensors, each of shape (B, D_i) where D_i is the
                         dimension of the i-th WSI encoder. Length must match num_sources.
            normalize: Whether to apply L2 normalization to each projected feature.
            
        Returns:
            Tensor of shape (B, num_sources, llm_hidden_size) containing all projected features.
        """
        if len(wsi_features) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} WSI features, got {len(wsi_features)}"
            )
        
        projected = []
        for i, (feat, projector, norm) in enumerate(
                zip(wsi_features, self.projectors, self.post_norms)):
            # Validate input dimension
            if feat.size(-1) != self.wsi_input_dims[i]:
                raise ValueError(
                    f"WSI source {i}: expected dim {self.wsi_input_dims[i]}, got {feat.size(-1)}"
                )
            
            # Project
            proj = projector(feat)  # (B, llm_hidden_size)
            proj = norm(proj)
            
            # Optional normalization
            if normalize:
                proj = nn.functional.normalize(proj, p=2, dim=-1)
            
            # Apply dropout
            proj = self.dropout(proj)
            
            projected.append(proj)
        
        # Stack along source dimension: (B, num_sources, llm_hidden_size)
        return torch.stack(projected, dim=1)
    
    def forward_single(self, wsi_feature: torch.Tensor, source_idx: int) -> torch.Tensor:
        """
        Project a single WSI feature from a specific source.
        
        Args:
            wsi_feature: Tensor of shape (B, D_i) or (D_i,)
            source_idx: Index of the WSI source
            
        Returns:
            Projected tensor of shape (B, llm_hidden_size) or (llm_hidden_size,)
        """
        if source_idx < 0 or source_idx >= self.num_sources:
            raise ValueError(f"Invalid source_idx {source_idx}, must be in [0, {self.num_sources})")
        
        return self.dropout(self.post_norms[source_idx](self.projectors[source_idx](wsi_feature)))
    
    def get_output_sequence_length(self) -> int:
        """Return the number of WSI tokens that will be added to the sequence."""
        return self.num_sources
    
    def enable_input_require_grads(self):
        """Enable gradient computation for inputs (needed for gradient checkpointing)."""
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        for projector in self.projectors:
            # Handle both Sequential (MLP) and Linear (Single Layer)
            target = projector.layers[0] if isinstance(projector.layers, nn.Sequential) else projector.layers
            target.register_forward_hook(make_inputs_require_grad)
    
    def extra_repr(self) -> str:
        return (
            f"wsi_input_dims={self.wsi_input_dims}, "
            f"llm_hidden_size={self.llm_hidden_size}, "
            f"num_sources={self.num_sources}"
        )

# Copyright (c) OpenMMLab. All rights reserved.
"""Prompt-conditioned pathology adapter for Qwen3.5 text models."""

import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModel
from mmengine.utils import is_list_of
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AddedToken, AutoConfig, GenerationConfig, StoppingCriteriaList
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.utils import IMAGE_TOKEN_INDEX, IGNORE_INDEX, StopWordStoppingCriteria

from .custom_model import (
    InputComposer,
    MRoPEPositionIDGenerator,
    PromptConditionedPatchResampler,
    ROUTE_FAMILIES,
    RoutedLoRALinear,
    RegressionHead,
    SurvivalHead,
    WSIProjector,
    cox_ph_loss,
    logistic_hazard_loss,
    replace_linear_with_routed_lora,
)
from .modules import dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
def _safe_token_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    safe_ids = input_ids.clone()
    safe_ids[safe_ids < 0] = pad_token_id
    return safe_ids


def _strip_image_tokens(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    padding_side: str,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attn = attention_mask.bool()
    keep = attn & (input_ids != IMAGE_TOKEN_INDEX)
    batch_size, seq_len = input_ids.shape
    lengths = keep.sum(dim=1, keepdim=True)
    rank = keep.long().cumsum(dim=1) - 1
    if padding_side == 'right':
        dst = rank
    else:
        dst = seq_len - lengths + rank
    dst = dst.clamp(min=0, max=max(seq_len - 1, 0))

    dummy_col = torch.full_like(dst, seq_len)
    scatter_dst = torch.where(keep, dst, dummy_col)
    new_ids_ext = torch.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    new_mask_ext = torch.zeros((batch_size, seq_len + 1), dtype=torch.bool, device=input_ids.device)
    new_ids_ext.scatter_(1, scatter_dst, torch.where(keep, input_ids, torch.full_like(input_ids, pad_token_id)))
    new_mask_ext.scatter_(1, scatter_dst, keep)
    new_ids = new_ids_ext[:, :seq_len]
    new_mask = new_mask_ext[:, :seq_len]

    new_labels = None
    if labels is not None:
        new_labels_ext = torch.full((batch_size, seq_len + 1), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        new_labels_ext.scatter_(1, scatter_dst, torch.where(keep, labels, torch.full_like(labels, IGNORE_INDEX)))
        new_labels = new_labels_ext[:, :seq_len]
    return new_ids, new_labels, new_mask


def _linear_position_ids(attention_mask: torch.Tensor, rows: int = 4) -> torch.Tensor:
    valid = attention_mask.bool()
    pos = valid.long().cumsum(dim=1).sub_(1).clamp_min_(0)
    pos = pos.masked_fill(~valid, 0)
    return pos.unsqueeze(0).expand(rows, -1, -1).contiguous()


def _raise_if_nonfinite(name: str, value: torch.Tensor, extra: str = "") -> None:
    if torch.isfinite(value).all():
        return
    value32 = value.detach().float()
    finite = value32[torch.isfinite(value32)]
    if finite.numel() > 0:
        min_value = float(finite.min().item())
        max_value = float(finite.max().item())
    else:
        min_value = float('nan')
        max_value = float('nan')
    suffix = f" {extra}" if extra else ""
    raise RuntimeError(
        f"Non-finite tensor detected in {name}:{suffix} "
        f"shape={tuple(value.shape)} dtype={value.dtype} "
        f"min={min_value:.6g} max={max_value:.6g}"
    )


def _sample_keep_mask(
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    value = value.to(device=device).bool().view(-1)
    if value.numel() != batch_size:
        raise ValueError(
            f"sample keep mask must have {batch_size} entries, got {value.numel()}")
    return value


def _prepare_text_or_wsi_inputs(
    llm,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    wsi_embeddings: Optional[torch.Tensor],
    padding_side: str,
    wsi_sample_keep: Optional[torch.Tensor] = None,
):
    pad_token_id = getattr(llm.config, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(llm.config, 'eos_token_id', 0)
    input_ids, labels, attention_mask = _strip_image_tokens(
        input_ids, labels, attention_mask, padding_side, int(pad_token_id)
    )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask = attention_mask.bool()
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    token_embed = llm.get_input_embeddings()
    safe_ids = _safe_token_ids(input_ids, int(pad_token_id))
    text_embeds = token_embed(safe_ids)
    if wsi_embeddings is None or wsi_embeddings.numel() == 0:
        return {
            'input_ids': None,
            'inputs_embeds': text_embeds,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': _linear_position_ids(attention_mask),
        }

    batch_size = input_ids.size(0)
    wsi_sample_keep = _sample_keep_mask(wsi_sample_keep, batch_size, input_ids.device)

    embeds_list, labels_list, attention_list, pos_list = [], [], [], []
    for b_idx in range(batch_size):
        valid = attention_mask[b_idx]
        text = text_embeds[b_idx, valid]
        lbl = labels[b_idx, valid]
        if bool(wsi_sample_keep[b_idx].item()):
            wsi = wsi_embeddings[b_idx].to(device=text.device, dtype=text.dtype)
            cur_emb = torch.cat([wsi, text], dim=0)
            cur_lbl = torch.cat([
                torch.full((wsi.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=labels.device),
                lbl,
            ])
        else:
            cur_emb = text
            cur_lbl = lbl
        cur_attn = torch.ones(cur_emb.size(0), dtype=torch.bool, device=cur_emb.device)
        seq = torch.arange(cur_emb.size(0), device=cur_emb.device, dtype=torch.long)
        embeds_list.append(cur_emb)
        labels_list.append(cur_lbl)
        attention_list.append(cur_attn)
        pos_list.append(seq.unsqueeze(0).expand(4, -1))

    composed = InputComposer()(embeds_list, labels_list, attention_list, pos_list, padding_side, IGNORE_INDEX)
    return {'input_ids': None, **composed}


def prepare_inputs_labels_for_qwen3_5(
    llm,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_batch_indices: Optional[torch.Tensor] = None,
    vision_token_positions: Optional[torch.Tensor] = None,
    vision_token_valid: Optional[torch.Tensor] = None,
    wsi_embeddings: Optional[torch.Tensor] = None,
    patch_sample_keep: Optional[torch.Tensor] = None,
    wsi_sample_keep: Optional[torch.Tensor] = None,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    position_generator: Optional[MRoPEPositionIDGenerator] = None,
    patch_position_encoding: str = 'linear',
    composer: Optional[InputComposer] = None,
    padding_side: str = 'right',
    **kwargs,
):
    batch_size = input_ids.size(0)
    patch_sample_keep = _sample_keep_mask(patch_sample_keep, batch_size, input_ids.device)
    wsi_sample_keep = _sample_keep_mask(wsi_sample_keep, batch_size, input_ids.device)
    has_patch = pixel_values is not None and pixel_values.numel() > 0
    has_wsi = wsi_embeddings is not None and wsi_embeddings.numel() > 0
    if not has_patch:
        return _prepare_text_or_wsi_inputs(
            llm, input_ids, labels, attention_mask, wsi_embeddings, padding_side,
            wsi_sample_keep=wsi_sample_keep)

    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask = attention_mask.bool()
    patch_position_encoding = str(patch_position_encoding).lower()
    if patch_position_encoding not in ('linear', 'mrope'):
        raise ValueError("patch_position_encoding must be 'linear' or 'mrope'.")
    if position_generator is None:
        position_generator = MRoPEPositionIDGenerator()
    if composer is None:
        composer = InputComposer()
    if vision_start_token_id is None or vision_end_token_id is None:
        raise ValueError("Qwen3.5 patch injection requires vision_start_token_id and vision_end_token_id.")

    token_embed = llm.get_input_embeddings()
    pad_token_id = getattr(llm.config, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(llm.config, 'eos_token_id', 0)
    device = input_ids.device
    dtype = token_embed.weight.dtype
    hidden_dim = token_embed.weight.shape[1]
    special_ids = torch.tensor([vision_start_token_id, vision_end_token_id], device=device, dtype=torch.long)
    vision_start_embed, vision_end_embed = token_embed(special_ids).view(2, 1, hidden_dim)

    payloads_by_sample = [[] for _ in range(batch_size)]
    if image_batch_indices is None:
        image_batch_indices = torch.arange(pixel_values.size(0), device=device).clamp(max=batch_size - 1)
    for img_idx, b_idx in enumerate(image_batch_indices.to(device=device).tolist()):
        payloads_by_sample[int(b_idx)].append((
            pixel_values[img_idx],
            vision_token_positions[img_idx],
            vision_token_valid[img_idx],
        ))

    embeds_list, labels_list, attention_list, positions_list = [], [], [], []
    patch_spans_by_sample = []
    for b_idx in range(batch_size):
        cur_ids = input_ids[b_idx, attention_mask[b_idx]]
        cur_labels = labels[b_idx, attention_mask[b_idx]]
        image_positions = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        boundaries = [-1] + image_positions + [cur_ids.numel()]
        payload_idx = 0
        keep_patch = bool(patch_sample_keep[b_idx].item())
        keep_wsi = has_wsi and bool(wsi_sample_keep[b_idx].item())
        wsi_inserted = False
        pieces_embeds, pieces_labels, pieces_attn, patch_blocks = [], [], [], []

        for seg_idx in range(len(boundaries) - 1):
            text_start = boundaries[seg_idx] + 1
            text_end = boundaries[seg_idx + 1]
            text_ids = cur_ids[text_start:text_end]
            text_labels = cur_labels[text_start:text_end]
            if text_ids.numel() > 0:
                safe_ids = _safe_token_ids(text_ids, int(pad_token_id))
                text_embeds = token_embed(safe_ids)
                pieces_embeds.append(text_embeds)
                pieces_labels.append(text_labels)
                pieces_attn.append(torch.ones(text_embeds.size(0), dtype=torch.bool, device=device))

            is_image_slot = seg_idx < len(boundaries) - 2
            if is_image_slot:
                if keep_wsi and not wsi_inserted:
                    cur_wsi = wsi_embeddings[b_idx].to(device=device, dtype=dtype)
                    pieces_embeds.append(cur_wsi)
                    pieces_labels.append(torch.full((cur_wsi.size(0),), IGNORE_INDEX,
                                                    dtype=labels.dtype, device=device))
                    pieces_attn.append(torch.ones(cur_wsi.size(0), dtype=torch.bool, device=device))
                    wsi_inserted = True

                if not keep_patch:
                    if payload_idx < len(payloads_by_sample[b_idx]):
                        payload_idx += 1
                    continue

                if payload_idx >= len(payloads_by_sample[b_idx]):
                    raise ValueError(f"Sample {b_idx} has more <image> tokens than visual feature groups.")
                patch_embeds, patch_positions, patch_valid = payloads_by_sample[b_idx][payload_idx]
                payload_idx += 1
                patch_embeds = patch_embeds.to(device=device, dtype=dtype)
                patch_positions = patch_positions.to(device=device)
                patch_valid = patch_valid.to(device=device).bool()

                current_offset = sum(x.size(0) for x in pieces_embeds)
                pieces_embeds.extend([vision_start_embed, patch_embeds, vision_end_embed])
                pieces_labels.extend([
                    torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=device),
                    torch.full((patch_embeds.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=device),
                    torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=device),
                ])
                pieces_attn.extend([
                    torch.ones(1, dtype=torch.bool, device=device),
                    patch_valid,
                    torch.ones(1, dtype=torch.bool, device=device),
                ])
                patch_blocks.append((current_offset + 1, patch_positions, patch_valid,
                                     current_offset + 1 + patch_embeds.size(0)))

        if payload_idx != len(payloads_by_sample[b_idx]):
            raise ValueError(f"Sample {b_idx} has unused visual feature groups.")
        if keep_wsi and not wsi_inserted:
            cur_wsi = wsi_embeddings[b_idx].to(device=device, dtype=dtype)
            pieces_embeds = [cur_wsi] + pieces_embeds
            pieces_labels = [torch.full((cur_wsi.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=device)] + pieces_labels
            pieces_attn = [torch.ones(cur_wsi.size(0), dtype=torch.bool, device=device)] + pieces_attn
            patch_blocks = [
                (start + cur_wsi.size(0), pos, valid, end + cur_wsi.size(0))
                for start, pos, valid, end in patch_blocks
            ]

        if not pieces_embeds:
            pieces_embeds.append(token_embed(torch.tensor([int(pad_token_id)], device=device)))
            pieces_labels.append(torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=device))
            pieces_attn.append(torch.zeros(1, dtype=torch.bool, device=device))

        cur_embeds = torch.cat(pieces_embeds, dim=0)
        cur_labels = torch.cat(pieces_labels, dim=0)
        cur_attn = torch.cat(pieces_attn, dim=0)
        seq_pos = torch.arange(cur_embeds.size(0), device=device, dtype=torch.long)
        cur_pos = seq_pos.unsqueeze(0).expand(4, -1).clone()
        if patch_position_encoding == 'mrope':
            for patch_start, patch_positions, patch_valid, vision_end_idx in patch_blocks:
                cur_pos = position_generator(
                    seq_pos,
                    patch_start=patch_start,
                    token_positions=patch_positions,
                    token_valid=patch_valid,
                    vision_end_index=vision_end_idx,
                    base_position_ids=cur_pos,
                )
        embeds_list.append(cur_embeds)
        labels_list.append(cur_labels)
        attention_list.append(cur_attn)
        positions_list.append(cur_pos)
        patch_spans_by_sample.append([(int(start), int(end)) for start, _, _, end in patch_blocks])

    composed = composer(embeds_list, labels_list, attention_list, positions_list, padding_side, IGNORE_INDEX)
    max_spans = max((len(spans) for spans in patch_spans_by_sample), default=0)
    if max_spans > 0:
        span_tensor = torch.full((batch_size, max_spans, 2), -1, dtype=torch.long, device=device)
        max_len = composed['attention_mask'].size(1)
        for b_idx, spans in enumerate(patch_spans_by_sample):
            pad_offset = max_len - embeds_list[b_idx].size(0) if padding_side == 'left' else 0
            for span_idx, (start, end) in enumerate(spans):
                span_tensor[b_idx, span_idx, 0] = start + pad_offset
                span_tensor[b_idx, span_idx, 1] = end + pad_offset
        composed['vision_token_spans'] = span_tensor
    return {'input_ids': None, **composed}
class LLaVAModel_qwen3_5(BaseModel):
    """Qwen3.5 multimodal model with prompt-conditioned patch resampling."""

    SUPPORT_CONFIGS = {
        'SDPA': ('LlamaConfig', 'GemmaConfig', 'MistralConfig', 'MixtralConfig',
                 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config',
                 'Qwen3Config', 'Qwen3_5Config'),
        'FLASH2': ('InternLM2Config', 'LlamaConfig', 'GemmaConfig', 'MistralConfig',
                   'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config',
                   'Phi3Config', 'Qwen3Config', 'Qwen3_5Config'),
    }

    def __init__(
        self,
        llm,
        tokenizer,
        freeze_llm: bool = True,
        pretrained_pth: Optional[str] = None,
        pretrained_patch_resampler: Optional[str] = None,
        llm_lora: Optional[Dict] = None,
        use_activation_checkpointing: bool = True,
        max_position_embeddings: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
        stop_words: Optional[List[str]] = None,
        enable_regression: bool = True,
        reg_token: str = '<REG>',
        enable_survival: bool = True,
        srv_token: str = '<SRV>',
        num_survival_intervals: int = 6,
        survival_method: str = 'cox',
        gen_forcing: bool = True,
        lambda_llm: float = 0.1,
        lambda_reg: float = 1.0,
        lambda_srv: float = 1.0,
        prompt_resampler_cfg: Optional[Dict] = None,
        prompt_context_mode: str = 'llm_hidden',
        prompt_context_layer: Union[int, str] = 'auto',
        prompt_context_detach: bool = True,
        patch_position_encoding: str = 'linear',
        enable_nonfinite_checks: bool = False,
        wsi_feature_dims: Optional[List[int]] = None,
        wsi_dropout: float = 0.1,
        survival_head_dropout: float = 0.3,
        head_scaling: Union[float, List[float]] = (0.0, 0.0, 0.5),
        patch_modality_dropout: float = 0.0,
        wsi_modality_dropout: float = 0.0,
        modality_dropout_allow_text_only: bool = False,
        force_drop_patch: bool = False,
        force_drop_wsi: bool = False,
        trainable_module_prefixes: Optional[List[str]] = None,
        save_attention_heatmap: bool = False,
        attention_heatmap_dir: Optional[str] = None,
        save_patch_attention_h5: bool = False,
        patch_attention_h5_dir: Optional[str] = None,
        patch_attention_h5_dtype: str = 'float16',
        route_families: Optional[List[str]] = None,
        routed_lora_family_rank: Optional[int] = None,
        routed_lora_family_alpha: Optional[float] = None,
        routed_lora_trainable: str = 'all',
        freeze_patch_route_residual: bool = False,
    ):
        super().__init__()
        self.route_families = tuple(route_families or ROUTE_FAMILIES)
        if not self.route_families or len(set(self.route_families)) != len(self.route_families):
            raise ValueError(
                'route_families must contain unique non-empty family names, '
                f'got {self.route_families!r}.')
        self._route_family_context = None
        self.routed_lora_family_rank = routed_lora_family_rank
        self.routed_lora_family_alpha = routed_lora_family_alpha
        self.routed_lora_trainable = str(routed_lora_trainable).lower()
        if self.routed_lora_trainable not in ('all', 'family_only', 'shared_only'):
            raise ValueError(
                "routed_lora_trainable must be 'all', 'family_only', or "
                f"'shared_only', got {routed_lora_trainable!r}.")
        self.freeze_patch_route_residual = bool(freeze_patch_route_residual)
        self.freeze_llm = freeze_llm
        self.enable_regression = enable_regression
        self.enable_survival = enable_survival
        self.reg_token = reg_token
        self.srv_token = srv_token
        self.num_survival_intervals = num_survival_intervals
        self.survival_method = survival_method
        self.gen_forcing = gen_forcing
        self.lambda_llm = lambda_llm
        self.lambda_reg = lambda_reg
        self.lambda_srv = lambda_srv
        self.prompt_resampler_cfg = prompt_resampler_cfg
        self.prompt_context_mode = prompt_context_mode
        self.prompt_context_layer = prompt_context_layer if str(prompt_context_layer).lower() == 'auto' else int(prompt_context_layer)
        self.prompt_context_detach = bool(prompt_context_detach)
        self.patch_position_encoding = str(patch_position_encoding).lower()
        self.enable_nonfinite_checks = bool(enable_nonfinite_checks)
        self.enable_vision = prompt_resampler_cfg is not None
        self.wsi_feature_dims = wsi_feature_dims
        self.enable_wsi_injection = wsi_feature_dims is not None and len(wsi_feature_dims) > 0
        self.wsi_dropout = float(wsi_dropout)
        self.survival_head_dropout = float(survival_head_dropout)
        self.patch_modality_dropout = float(patch_modality_dropout)
        self.wsi_modality_dropout = float(wsi_modality_dropout)
        self.modality_dropout_allow_text_only = bool(modality_dropout_allow_text_only)
        self.force_drop_patch = bool(force_drop_patch)
        self.force_drop_wsi = bool(force_drop_wsi)
        self.trainable_module_prefixes = trainable_module_prefixes
        self.save_attention_heatmap = bool(save_attention_heatmap)
        self.attention_heatmap_dir = attention_heatmap_dir
        self.save_patch_attention_h5 = bool(save_patch_attention_h5)
        self.patch_attention_h5_dir = patch_attention_h5_dir
        self.patch_attention_h5_dtype = str(patch_attention_h5_dtype)
        if self.patch_attention_h5_dtype not in ('float16', 'float32'):
            raise ValueError("patch_attention_h5_dtype must be 'float16' or 'float32'.")
        self.use_llm_lora = llm_lora is not None
        self._use_llm_lora = self.use_llm_lora
        self.routed_lora_enabled = False
        self.reg_token_id = None
        self.srv_token_id = None
        self._last_hidden_state = None
        self._last_patch_attention = None
        self._last_patch_valid_mask = None
        self.is_first_iter = True
        if isinstance(head_scaling, (int, float)):
            self.head_scaling = [float(head_scaling)] * 3
        else:
            self.head_scaling = [float(x) for x in head_scaling]
            if len(self.head_scaling) != 3:
                raise ValueError("head_scaling must be a float or a length-3 list.")
        if self.prompt_context_mode not in ('embedding', 'llm_hidden'):
            raise ValueError("prompt_context_mode must be 'embedding' or 'llm_hidden'.")
        if self.patch_position_encoding not in ('linear', 'mrope'):
            raise ValueError("patch_position_encoding must be 'linear' or 'mrope'.")
        for name, value in (
            ('patch_modality_dropout', self.patch_modality_dropout),
            ('wsi_modality_dropout', self.wsi_modality_dropout),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        self._init_llm(llm, max_position_embeddings)
        self._setup_tokenizer_and_tokens(tokenizer)
        if self.enable_vision:
            self._init_prompt_resampler()
            if self.freeze_patch_route_residual:
                self.patch_resampler.family_query_residual.requires_grad_(False)
        if self.enable_wsi_injection:
            self._init_wsi_projector()
        if self.enable_regression or self.enable_survival:
            self._init_prediction_modules()
            self._init_special_lm_head()
        self._configure_training(llm_lora, use_activation_checkpointing)
        self._setup_generation(generation_kwargs, stop_words)
        self.position_generator = MRoPEPositionIDGenerator()
        self.input_composer = InputComposer()
        if pretrained_pth:
            self.load_state_dict(guess_load_checkpoint(pretrained_pth), strict=False)
        if pretrained_patch_resampler:
            self._load_pretrained_patch_resampler(pretrained_patch_resampler)
        self._apply_trainable_module_filter()
        self._apply_routed_lora_trainable_filter()

    def _init_llm(self, llm, max_position_embeddings: Optional[int]) -> None:
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(ConfigDict(llm), max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm)
        if hasattr(self.llm.config, 'use_cache'):
            self.llm.config.use_cache = False
        if hasattr(self.llm.config, 'text_config'):
            self.llm.config.text_config.use_cache = False
        dispatch_modules(self.llm)

    def _get_llm_hidden_size(self) -> int:
        if hasattr(self.llm.config, 'hidden_size'):
            return int(self.llm.config.hidden_size)
        if hasattr(self.llm.config, 'text_config'):
            return int(self.llm.config.text_config.hidden_size)
        raise AttributeError("Could not determine hidden_size from LLM config.")

    def _get_language_model_norm(self):
        paths_to_try = [
            lambda: self.llm.model.norm,
            lambda: self.llm.model.language_model.norm,
            lambda: self.llm.base_model.model.model.norm,
            lambda: self.llm.base_model.model.model.language_model.norm,
        ]
        for get_norm in paths_to_try:
            try:
                norm = get_norm()
                if norm is not None:
                    return norm
            except (AttributeError, TypeError):
                continue
        return None

    def _get_prompt_context_model(self):
        paths_to_try = [
            lambda: self.llm.model.language_model,
            lambda: self.llm.base_model.model.model.language_model,
            lambda: self.llm.base_model.model.language_model,
            lambda: self.llm.model,
            lambda: self.llm.base_model.model.model,
            lambda: self.llm.base_model.model,
        ]
        for get_model in paths_to_try:
            try:
                model = get_model()
                if model is not None:
                    return model
            except (AttributeError, TypeError):
                continue
        return self.llm

    def _run_prompt_context_model(self, model, **kwargs):
        kwargs.setdefault('use_cache', False)
        with self._temporary_attn_implementation('sdpa'):
            if self.prompt_context_detach:
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        return model(**kwargs)
                finally:
                    model.train(was_training)
            return model(**kwargs)

    def _setup_tokenizer_and_tokens(self, tokenizer) -> None:
        if isinstance(tokenizer, (dict, Config, ConfigDict)):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        self._original_vocab_size = len(self.tokenizer)
        added_tokens = []
        if self.enable_regression:
            added_tokens.append(self.reg_token)
        if self.enable_survival:
            added_tokens.append(self.srv_token)
        if added_tokens:
            self._add_special_tokens(added_tokens)
        if self.enable_regression:
            self.reg_token_id = self.tokenizer.convert_tokens_to_ids(self.reg_token)
            self._init_token_embedding(self.reg_token_id, 'regression')
        if self.enable_survival:
            self.srv_token_id = self.tokenizer.convert_tokens_to_ids(self.srv_token)
            self._init_token_embedding(self.srv_token_id, 'survival')

        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        if self.vision_start_token_id is None or self.vision_start_token_id < 0:
            self.vision_start_token_id = getattr(self.llm.config, 'vision_start_token_id', None)
        if self.vision_end_token_id is None or self.vision_end_token_id < 0:
            self.vision_end_token_id = getattr(self.llm.config, 'vision_end_token_id', None)

        if getattr(self.llm.config, 'pad_token_id', None) is None:
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._register_embedding_grad_hook()

    def _add_special_tokens(self, tokens: List[str]) -> None:
        added = [AddedToken(t, normalized=False, special=True) for t in tokens]
        try:
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': added})
        except Exception:
            num_added = self.tokenizer.add_tokens(added, special_tokens=True)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            if hasattr(self.llm, 'tie_weights'):
                self.llm.tie_weights()

    def _init_token_embedding(self, token_id: int, task_type: str) -> None:
        emb = self.llm.get_input_embeddings()
        out = self.llm.get_output_embeddings()
        if emb is None or token_id is None or token_id >= emb.weight.size(0):
            return
        init_words = ["value", "number", "score"] if task_type == 'regression' else ["survival", "risk", "hazard"]
        valid_ids = []
        for word in init_words:
            tid = self.tokenizer.convert_tokens_to_ids(word)
            if tid is not None and 0 <= tid < emb.weight.size(0):
                valid_ids.append(tid)
        with torch.no_grad():
            base = emb.weight[valid_ids].mean(dim=0) if valid_ids else emb.weight.mean(dim=0)
            emb.weight[token_id] = base + 1e-3 * torch.randn_like(base)
            if out is not None and out is not emb and hasattr(out, 'weight') and token_id < out.weight.size(0):
                out.weight[token_id] = out.weight[valid_ids].mean(dim=0) if valid_ids else out.weight.mean(dim=0)

    def _register_embedding_grad_hook(self) -> None:
        old_vocab_size = self._original_vocab_size

        def _get_zero_hook(_name: str):
            def _zero_old_token_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None:
                    grad[:old_vocab_size] = 0
                return grad
            return _zero_old_token_grad

        emb = self.llm.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            if hasattr(self, '_embedding_grad_hook_handle'):
                self._embedding_grad_hook_handle.remove()
            self._embedding_grad_hook_handle = emb.weight.register_hook(_get_zero_hook('input'))

        is_tied = getattr(self.llm.config, 'tie_word_embeddings', True)
        if not is_tied:
            out = self.llm.get_output_embeddings()
            if out is not None and hasattr(out, 'weight'):
                if hasattr(self, '_output_grad_hook_handle'):
                    self._output_grad_hook_handle.remove()
                self._output_grad_hook_handle = out.weight.register_hook(_get_zero_hook('output'))

    def _estimate_embedding_rms(self) -> Optional[float]:
        emb = self.llm.get_input_embeddings()
        if emb is None or not hasattr(emb, 'weight'):
            return None
        with torch.no_grad():
            return float(emb.weight.detach().float().pow(2).mean(dim=-1).sqrt().mean().item())

    def _init_prompt_resampler(self) -> None:
        cfg = dict(
            patch_dim=768,
            llm_hidden_size=self._get_llm_hidden_size(),
            resampler_dim=min(2048, self._get_llm_hidden_size()),
            num_query=8,
            num_layers=2,
            num_heads=8,
            dropout=0.0,
            use_local_conv=True,
            query_init_std=0.5,
        )
        cfg.update(self.prompt_resampler_cfg or {})
        cfg['llm_hidden_size'] = self._get_llm_hidden_size()
        cfg['route_families'] = self.route_families
        self.patch_resampler = PromptConditionedPatchResampler(**cfg)
        self.patch_resampler.set_output_rms(self._estimate_embedding_rms())
        print_log(f"[PromptResampler] cfg={cfg}", 'current')
    def _init_wsi_projector(self) -> None:
        self.wsi_projector = WSIProjector(
            wsi_input_dims=self.wsi_feature_dims,
            llm_hidden_size=self._get_llm_hidden_size(),
            hidden_mult=self.head_scaling[2],
            dropout=self.wsi_dropout,
        )

    def _init_prediction_modules(self) -> None:
        hidden = self._get_llm_hidden_size()
        if self.enable_regression:
            self.regression_head = RegressionHead(hidden, hidden_mult=self.head_scaling[0])
            self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0)
        if self.enable_survival:
            self.survival_head = SurvivalHead(
                hidden,
                method=self.survival_method,
                num_intervals=self.num_survival_intervals,
                hidden_mult=self.head_scaling[1],
                dropout=self.survival_head_dropout,
            )
            self.survival_loss_fn = cox_ph_loss if self.survival_method == 'cox' else logistic_hazard_loss

    def _task_token_ids(self) -> List[int]:
        token_ids = []
        if self.enable_regression and self.reg_token_id is not None:
            token_ids.append(int(self.reg_token_id))
        if self.enable_survival and self.srv_token_id is not None:
            token_ids.append(int(self.srv_token_id))
        return token_ids

    def _init_special_lm_head(self) -> None:
        token_ids = self._task_token_ids()
        if not token_ids:
            self.special_lm_token_ids = []
            self.special_lm_head = None
            return

        hidden = self._get_llm_hidden_size()
        self.special_lm_token_ids = token_ids
        self.special_lm_head = nn.Linear(hidden, len(token_ids), bias=True)

        out = self.llm.get_output_embeddings()
        if out is not None and hasattr(out, 'weight'):
            with torch.no_grad():
                weight = out.weight[token_ids].detach().float()
                self.special_lm_head.weight.copy_(weight.to(dtype=self.special_lm_head.weight.dtype))
                self.special_lm_head.bias.zero_()

    def _apply_special_lm_logits(
        self,
        active_logits: torch.Tensor,
        active_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if not getattr(self, 'special_lm_token_ids', None) or self.special_lm_head is None:
            return active_logits
        token_ids = torch.as_tensor(self.special_lm_token_ids, device=active_logits.device, dtype=torch.long)
        special_logits = self.special_lm_head(
            active_hidden.to(dtype=next(self.special_lm_head.parameters()).dtype)
        )
        active_logits = active_logits.to(dtype=special_logits.dtype).clone()
        active_logits.index_copy_(1, token_ids, special_logits)
        return active_logits

    def _maybe_raise_if_nonfinite(self, name: str, value: torch.Tensor, extra: str = "") -> None:
        if not self.enable_nonfinite_checks:
            return
        _raise_if_nonfinite(name, value, extra=extra)

    @contextmanager
    def _temporary_attn_implementation(self, implementation: Optional[str]):
        """Temporarily override runtime attention backend for fragile eval paths."""
        if implementation is None:
            yield
            return

        targets = []
        llm_config = getattr(self.llm, 'config', None)
        configs = [llm_config, getattr(llm_config, 'text_config', None)]
        try:
            prompt_config = getattr(self._get_prompt_context_model(), 'config', None)
            configs.extend([prompt_config, getattr(prompt_config, 'text_config', None)])
        except Exception:
            pass

        seen_configs = set()
        for cfg in configs:
            if cfg is None or id(cfg) in seen_configs:
                continue
            seen_configs.add(id(cfg))
            old_values = {}
            for attr in ('_attn_implementation', 'attn_implementation'):
                if hasattr(cfg, attr):
                    old_values[attr] = getattr(cfg, attr)
                    setattr(cfg, attr, implementation)
            if old_values:
                targets.append((cfg, old_values))

        try:
            yield
        finally:
            for cfg, old_values in targets:
                for attr, old_value in old_values.items():
                    setattr(cfg, attr, old_value)

    def _resolve_prompt_context_layer(self, hidden_states) -> int:
        if str(self.prompt_context_layer).lower() == 'auto':
            num_llm_layers = max(len(hidden_states) - 1, 1)
            return min(len(hidden_states) - 1, max(1, round(num_llm_layers * 2 / 3)))
        layer = int(self.prompt_context_layer)
        if not (-len(hidden_states) <= layer < len(hidden_states)):
            raise IndexError(
                f"prompt_context_layer={layer} is out of range for "
                f"{len(hidden_states)} hidden-state tensors."
            )
        return layer if layer >= 0 else len(hidden_states) + layer

    def _prompt_mask(self, data: Dict[str, Any]) -> torch.Tensor:
        input_ids = data['input_ids']
        labels = data.get('labels')
        attention_mask = data.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prompt_mask = attention_mask.bool() & (input_ids != IMAGE_TOKEN_INDEX)
        if labels is not None:
            prompt_mask = prompt_mask & (labels == IGNORE_INDEX)
        empty = ~prompt_mask.any(dim=1)
        fallback_mask = attention_mask.bool() & (input_ids != IMAGE_TOKEN_INDEX)
        fallback_idx = fallback_mask.long().argmax(dim=1, keepdim=True)
        fallback_one = torch.zeros_like(prompt_mask)
        fallback_one.scatter_(1, fallback_idx, True)
        prompt_mask = torch.where(empty.unsqueeze(1), fallback_one, prompt_mask)
        return prompt_mask

    def _text_context_embeds(
        self,
        input_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_token_id = self.llm.config.pad_token_id or self.tokenizer.eos_token_id or 0
        safe_ids = _safe_token_ids(input_ids, int(pad_token_id))
        token_embed = self.llm.get_input_embeddings()
        inputs_embeds = token_embed(safe_ids)
        embeds_list, labels_list, attention_list, pos_list = [], [], [], []
        for b_idx in range(input_ids.size(0)):
            cur_mask = prompt_mask[b_idx]
            cur_embeds = inputs_embeds[b_idx, cur_mask]
            cur_len = cur_embeds.size(0)
            cur_attn = torch.ones(cur_len, dtype=torch.bool, device=cur_embeds.device)
            cur_pos = torch.arange(cur_len, device=cur_embeds.device, dtype=torch.long)
            embeds_list.append(cur_embeds)
            labels_list.append(torch.full((cur_len,), IGNORE_INDEX, dtype=torch.long, device=cur_embeds.device))
            attention_list.append(cur_attn)
            pos_list.append(cur_pos.unsqueeze(0).expand(4, -1))
        prompt_inputs = self.input_composer(
            embeds_list,
            labels_list,
            attention_list,
            pos_list,
            'left',
            IGNORE_INDEX,
        )

        if self.prompt_context_mode == 'embedding':
            return prompt_inputs['inputs_embeds'], prompt_inputs['attention_mask']

        prompt_model = self._get_prompt_context_model()
        outputs = self._run_prompt_context_model(
            prompt_model,
            inputs_embeds=prompt_inputs['inputs_embeds'],
            attention_mask=prompt_inputs['attention_mask'],
            position_ids=prompt_inputs['position_ids'],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise TypeError(f"Prompt context model did not return hidden_states: {type(outputs)!r}")
        selected_layer = self._resolve_prompt_context_layer(hidden_states)
        context_embeds = hidden_states[selected_layer]
        if selected_layer == len(hidden_states) - 1:
            final_norm = self._get_language_model_norm()
            if final_norm is not None:
                context_embeds = final_norm(context_embeds)
        self._maybe_raise_if_nonfinite(
            'prompt_context_embeds',
            context_embeds,
            extra=f"mode={self.prompt_context_mode} layer={self.prompt_context_layer}",
        )
        if self.prompt_context_detach:
            context_embeds = context_embeds.detach()
        return context_embeds, prompt_inputs['attention_mask']

    def _prompt_embeds_for_images(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = data['input_ids']
        prompt_mask = self._prompt_mask(data)
        prompt_embeds, prompt_mask = self._text_context_embeds(input_ids, prompt_mask)
        image_batch_indices = data.get('image_batch_indices')
        if image_batch_indices is None:
            image_batch_indices = torch.arange(data['features'].size(0), device=input_ids.device).clamp(max=input_ids.size(0) - 1)
        image_batch_indices = image_batch_indices.to(device=input_ids.device, dtype=torch.long)
        return prompt_embeds[image_batch_indices], prompt_mask[image_batch_indices]

    def _normalize_route_family(self, route_family: Any, batch_size: int,
                                device: torch.device) -> torch.Tensor:
        """Validate and encode the required per-sample route field."""
        if route_family is None:
            raise ValueError(
                'Missing route_family. Every sample must provide one of '
                f'{self.route_families!r}; route is never inferred from the prompt.')
        if torch.is_tensor(route_family):
            if route_family.dtype.is_floating_point:
                raise ValueError('route_family tensor must contain integer ids, not floats.')
            route_ids = route_family.to(device=device, dtype=torch.long).view(-1)
        else:
            if isinstance(route_family, str):
                route_family = [route_family]
            try:
                route_values = list(route_family)
            except TypeError as exc:
                raise ValueError('route_family must be a per-sample sequence.') from exc
            if len(route_values) != batch_size:
                raise ValueError(
                    f'route_family must have {batch_size} entries, got {len(route_values)}.')
            route_to_id = {family: idx for idx, family in enumerate(self.route_families)}
            invalid = [value for value in route_values if value not in route_to_id]
            if invalid:
                raise ValueError(
                    f'Invalid route_family values {invalid!r}; expected one of '
                    f'{self.route_families!r}.')
            route_ids = torch.tensor(
                [route_to_id[value] for value in route_values],
                device=device,
                dtype=torch.long,
            )
        if route_ids.numel() != batch_size:
            raise ValueError(
                f'route_family must be per-sample with shape [{batch_size}], '
                f'got {tuple(route_ids.shape)}.')
        if bool(((route_ids < 0) | (route_ids >= len(self.route_families))).any().item()):
            raise ValueError(
                f'Invalid route_family ids {route_ids.tolist()}; expected '
                f'0..{len(self.route_families) - 1}.')
        return route_ids

    @contextmanager
    def _routed_lora_context(self, route_family: Optional[torch.Tensor]):
        """Install one route for all LoRA calls in a forward/generation pass."""
        if not self.routed_lora_enabled:
            yield
            return
        previous = []
        for module in self.llm.modules():
            if isinstance(module, RoutedLoRALinear):
                previous.append((module, module._route_family))
                module.set_route_family(route_family)
        if not previous:
            raise RuntimeError('routed_lora_enabled=True but no routed linear modules exist.')
        try:
            yield
        finally:
            # Gradient-checkpointed transformer blocks are recomputed during
            # backward, after this forward context has exited. Keep the route
            # installed until the next serial training forward overwrites it;
            # otherwise recomputation sees route=None. Eval/generation has no
            # deferred recomputation and can restore the previous context.
            keep_for_backward = (
                self.training
                and torch.is_grad_enabled()
                and bool(getattr(self.llm, 'is_gradient_checkpointing', False))
            )
            if not keep_for_backward:
                for module, old_route in previous:
                    module.set_route_family(old_route)

    def _make_modality_keep_masks(
        self,
        batch_size: int,
        has_patch: bool,
        has_wsi: bool,
        device: torch.device,
        mode: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        patch_keep = torch.ones(batch_size, dtype=torch.bool, device=device) if has_patch else None
        wsi_keep = torch.ones(batch_size, dtype=torch.bool, device=device) if has_wsi else None

        if patch_keep is not None and self.force_drop_patch:
            patch_keep.zero_()
        if wsi_keep is not None and self.force_drop_wsi:
            wsi_keep.zero_()

        if mode == 'loss' and self.training:
            if patch_keep is not None and not self.force_drop_patch and self.patch_modality_dropout > 0:
                patch_keep &= torch.rand(batch_size, device=device) >= self.patch_modality_dropout
            if wsi_keep is not None and not self.force_drop_wsi and self.wsi_modality_dropout > 0:
                wsi_keep &= torch.rand(batch_size, device=device) >= self.wsi_modality_dropout

        if (
            not self.modality_dropout_allow_text_only
            and patch_keep is not None
            and wsi_keep is not None
        ):
            both_dropped = ~patch_keep & ~wsi_keep
            if both_dropped.any():
                can_restore_patch = not self.force_drop_patch
                can_restore_wsi = not self.force_drop_wsi
                if can_restore_patch and can_restore_wsi:
                    restore_patch = torch.rand(batch_size, device=device) < 0.5
                    restore_patch &= both_dropped
                    patch_keep |= restore_patch
                    wsi_keep |= both_dropped & ~restore_patch
                elif can_restore_patch:
                    patch_keep |= both_dropped
                elif can_restore_wsi:
                    wsi_keep |= both_dropped

        return patch_keep, wsi_keep

    def _project_vision_features(
        self,
        data: Dict[str, Any],
        route_family: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.enable_vision:
            raise RuntimeError("Received patch features but prompt_resampler_cfg is None.")
        param = next(self.patch_resampler.parameters())
        features = data['features'].to(device=param.device, dtype=param.dtype)
        feature_shapes = data.get('feature_shapes')
        if feature_shapes is not None:
            feature_shapes = feature_shapes.to(device=param.device)
        prompt_embeds, prompt_mask = self._prompt_embeds_for_images(data)
        prompt_embeds = prompt_embeds.to(device=param.device, dtype=param.dtype)
        prompt_mask = prompt_mask.to(device=param.device)
        image_batch_indices = data.get('image_batch_indices')
        if image_batch_indices is None:
            image_batch_indices = torch.arange(
                features.size(0), device=route_family.device, dtype=torch.long)
        image_batch_indices = image_batch_indices.to(
            device=route_family.device, dtype=torch.long).view(-1)
        if image_batch_indices.numel() != features.size(0):
            raise ValueError(
                'image_batch_indices must contain one sample index per image: '
                f'{image_batch_indices.numel()} vs {features.size(0)}.')
        image_route_family = route_family.index_select(0, image_batch_indices)
        out = self.patch_resampler(
            features=features,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            feature_shapes=feature_shapes,
            route_family=image_route_family,
        )
        self._last_patch_attention = out['patch_attention'].detach()
        self._last_patch_valid_mask = out['patch_valid_mask'].detach()
        return {
            'pixel_values': out['visual_tokens'],
            'visual_tokens': out['visual_tokens'],
            'vision_token_positions': out['token_positions'],
            'vision_token_valid': out['token_valid'],
            'patch_attention': out['patch_attention'],
            'patch_valid_mask': out['patch_valid_mask'],
            'region_attention': out.get('region_attention'),
            'region_attention_heads': out.get('region_attention_heads'),
            'visual_to_region_attention': out.get('visual_to_region_attention'),
            'visual_to_region_attention_heads': out.get('visual_to_region_attention_heads'),
            'token_positions': out['token_positions'],
        }

    def _project_wsi_features(self, wsi_features: List[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.enable_wsi_injection or not hasattr(self, 'wsi_projector'):
            return None
        param = next(self.wsi_projector.parameters())
        batch_size = len(wsi_features)
        source_features = []
        for src_idx in range(len(wsi_features[0])):
            source_features.append(torch.stack([
                wsi_features[b][src_idx].to(device=param.device, dtype=param.dtype)
                for b in range(batch_size)
            ]))
        return self.wsi_projector(source_features)
    def forward(self, data: Dict[str, Any], data_samples: Optional[List] = None,
                mode: str = 'loss') -> Any:
        routed_data = dict(data)
        batch_size = int(routed_data['input_ids'].size(0))
        route_family = routed_data.pop('route_family', None)
        if route_family is None:
            route_family = routed_data.get('route_family_ids')
        route_family_ids = self._normalize_route_family(
            route_family, batch_size, routed_data['input_ids'].device)
        routed_data['route_family_ids'] = route_family_ids
        with self._routed_lora_context(route_family_ids):
            return self._forward_impl(routed_data, data_samples, mode)

    def _forward_impl(self, data: Dict[str, Any],
                      data_samples: Optional[List] = None,
                      mode: str = 'loss') -> Any:
        data = dict(data)
        if self.is_first_iter:
            first_tensor = next((v for v in data.values() if torch.is_tensor(v)), None)
            if first_tensor is not None:
                self.to(first_tensor.device)
            self.is_first_iter = False

        regression_targets = data.pop('regression_targets', None)
        survival_targets = data.pop('survival_targets', None)
        task_categories = data.get('category', None)
        has_visual = data.get('features') is not None
        has_wsi_features = self.enable_wsi_injection and data.get('wsi_features') is not None
        batch_size = int(data['input_ids'].size(0))
        mask_device = data['input_ids'].device
        patch_sample_keep, wsi_sample_keep = self._make_modality_keep_masks(
            batch_size=batch_size,
            has_patch=has_visual,
            has_wsi=has_wsi_features,
            device=mask_device,
            mode=mode,
        )

        attention_save_payload = None
        if has_visual:
            projected = self._project_vision_features(data, data['route_family_ids'])
            if mode == 'predict':
                self._save_attention_heatmaps(data, projected)
                attention_save_payload = self._build_patch_attention_payload(data, projected)
            data['pixel_values'] = projected['pixel_values']
            data['vision_token_positions'] = projected['vision_token_positions']
            data['vision_token_valid'] = projected['vision_token_valid']
            data['patch_sample_keep'] = patch_sample_keep
        else:
            data['pixel_values'] = None
            data['vision_token_positions'] = None
            data['vision_token_valid'] = None
            data['image_batch_indices'] = None
            data['patch_sample_keep'] = None
        data.pop('features', None)
        data.pop('feature_shapes', None)
        data.pop('feature_paths', None)

        wsi_embeddings = None
        if self.enable_wsi_injection and data.get('wsi_features') is not None:
            wsi_embeddings = self._project_wsi_features(data.pop('wsi_features'))
        else:
            data.pop('wsi_features', None)
        data['wsi_embeddings'] = wsi_embeddings
        data['wsi_sample_keep'] = wsi_sample_keep

        if mode == 'predict':
            for key in ('input_ids', 'labels', 'attention_mask'):
                if torch.is_tensor(data.get(key)):
                    data[key] = data[key].clone()
            self._strip_assistant_targets(data)
        padding_side = 'left'
        data = prepare_inputs_labels_for_qwen3_5(
            llm=self.llm,
            padding_side=padding_side,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            position_generator=self.position_generator,
            patch_position_encoding=self.patch_position_encoding,
            composer=self.input_composer,
            **data,
        )
        if attention_save_payload is not None:
            data['_patch_attention_save_payload'] = attention_save_payload
        if mode == 'loss':
            return self.compute_loss(data, data_samples, regression_targets, survival_targets)
        if mode == 'tensor':
            return self._forward(data, data_samples)
        if mode == 'predict':
            return self.predict(data, data_samples, regression_targets, survival_targets, task_categories)
        raise NotImplementedError(f"Unsupported mode: {mode}")
    def _forward(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None):
        kwargs = {k: data[k] for k in ['input_ids', 'inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
        return self.llm(**kwargs, use_cache=False, output_hidden_states=True, return_dict=True)

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Use the pre-weighted total loss for backward and keep components for logging only."""
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        if 'loss' in losses:
            loss = losses['loss']
        else:
            loss = sum(value for key, value in log_vars if 'loss' in key)

        log_vars_dict = OrderedDict()
        log_vars_dict['loss'] = loss
        for name, val in log_vars:
            if name != 'loss':
                log_vars_dict[name] = val

        return loss, log_vars_dict

    def compute_loss(
        self,
        data: Dict[str, torch.Tensor],
        data_samples: Optional[List] = None,
        regression_targets: Optional[torch.Tensor] = None,
        survival_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._forward(data, data_samples)
        logits = outputs.logits
        labels = data.get('labels')
        last_hidden = outputs.hidden_states[-1]
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_hidden = last_hidden[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            active = shift_labels != IGNORE_INDEX
            if active.any():
                active_logits = self._apply_special_lm_logits(
                    shift_logits[active],
                    shift_hidden[active],
                )
                lm_loss = nn.CrossEntropyLoss()(
                    active_logits.float(),
                    shift_labels[active],
                )
            else:
                lm_loss = shift_logits.new_zeros(())
        else:
            lm_loss = logits.new_zeros(())
        self._last_hidden_state = last_hidden
        reg_loss = self._compute_task_loss(labels, regression_targets, 'regression') if (
            self.enable_regression and regression_targets is not None and labels is not None
        ) else self._regularization_loss('regression', lm_loss)
        srv_loss = self._compute_task_loss(labels, survival_targets, 'survival') if (
            self.enable_survival and survival_targets is not None and labels is not None
        ) else self._regularization_loss('survival', lm_loss)
        self._maybe_raise_if_nonfinite('lm_loss', lm_loss)
        self._maybe_raise_if_nonfinite('reg_loss', reg_loss)
        self._maybe_raise_if_nonfinite('srv_loss', srv_loss)
        loss = self.lambda_llm * lm_loss + self.lambda_reg * reg_loss + self.lambda_srv * srv_loss
        return {
            'lm_loss': lm_loss.detach(),
            'reg_loss': reg_loss.detach(),
            'srv_loss': srv_loss.detach(),
            'loss': loss,
        }

    def _compute_task_loss(self, labels: torch.Tensor, targets: Union[torch.Tensor, Dict], task: str) -> torch.Tensor:
        token_id = self.reg_token_id if task == 'regression' else self.srv_token_id
        head = self.regression_head if task == 'regression' else self.survival_head
        token_mask = labels == token_id
        if token_id is None or not token_mask.any():
            return self._regularization_loss(task, labels.sum() * 0)
        seq = torch.arange(labels.size(1), device=labels.device).unsqueeze(0).expand_as(labels)
        last_pos = torch.where(token_mask, seq, torch.full_like(seq, -1)).max(dim=1).values
        valid = last_pos >= 0
        if not valid.any():
            return self._regularization_loss(task, labels.sum() * 0)
        b_idx = valid.nonzero(as_tuple=True)[0]
        task_embeds = self._last_hidden_state[b_idx, last_pos[valid]].to(dtype=next(head.parameters()).dtype)
        if task == 'regression':
            pred = self.regression_head(task_embeds).squeeze(-1)
            target = targets.to(device=pred.device, dtype=pred.dtype)[b_idx]
            keep = torch.isfinite(target)
            if not keep.any():
                return self._regularization_loss(task, pred.sum() * 0)
            return self.regression_loss_fn(pred[keep], target[keep])
        out = self.survival_head(task_embeds)
        if self.survival_method == 'discrete':
            target_y = targets['target_y'].to(device=out.device, dtype=out.dtype)[b_idx]
            at_risk = targets['at_risk_mask'].to(device=out.device, dtype=out.dtype)[b_idx]
            return logistic_hazard_loss(out, target_y, at_risk)
        time = targets['time'].to(device=out.device, dtype=out.dtype)[b_idx]
        event = targets['event'].to(device=out.device, dtype=out.dtype)[b_idx]
        keep = ~(torch.isnan(time) | torch.isnan(event))
        if not keep.any():
            return self._regularization_loss(task, out.sum() * 0)
        return cox_ph_loss(out[keep], time[keep], event[keep])

    def _regularization_loss(self, task: str, base: torch.Tensor, scale: float = 1e-6) -> torch.Tensor:
        module = None
        if task == 'regression' and hasattr(self, 'regression_head'):
            module = self.regression_head
        if task == 'survival' and hasattr(self, 'survival_head'):
            module = self.survival_head
        if module is None:
            return base * 0
        reg = sum((p.float() ** 2).mean() for p in module.parameters() if p.requires_grad)
        return reg.to(device=base.device, dtype=base.dtype) * scale if isinstance(reg, torch.Tensor) else base * 0

