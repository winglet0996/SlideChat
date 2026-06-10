import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
from transformers import InstructBlipQFormerConfig
from transformers.models.instructblip.modeling_instructblip import InstructBlipQFormerEncoder
from typing import Optional, Tuple, Iterable, Literal, Dict

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """Standard ConvNeXtV2 block (no partial-mask conv) for stable optimization."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class HighResConvNeXtV2Pyramid(nn.Module):
    """A concise ConvNeXtV2-style feature pyramid.

    Returns:
        stage_outputs: list of stage tensors
    """

    def __init__(self,
                 in_chans: int = 768,
                 depths: Iterable[int] = (2, 2, 4),
                 dims: Iterable[int] = (768, 1024, 1536),
                 drop_path_rate: float = 0.1,
                 num_downsamples: int = 2):
        super().__init__()

        depths = list(depths)
        dims = list(dims)
        self.dims = dims

        if len(depths) != len(dims):
            raise ValueError(f"depths and dims must have same length, got {len(depths)} and {len(dims)}")
        if num_downsamples > len(depths) - 1:
            raise ValueError(f"num_downsamples ({num_downsamples}) cannot exceed len(depths)-1 ({len(depths)-1})")

        self.input_proj = nn.Conv2d(in_chans, dims[0], kernel_size=1, bias=True) if in_chans != dims[0] else None

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = nn.ModuleList([ConvNeXtV2Block(dim=dim, drop_path=dp_rates[cur + j]) for j in range(depth)])
            self.stages.append(blocks)
            cur += depth

            if i < num_downsamples:
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dim, eps=1e-6, data_format='channels_first'),
                        nn.Conv2d(dim, dims[i + 1], kernel_size=2, stride=2, bias=True),
                    ))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if self.input_proj is not None:
            x = self.input_proj(x)

        stage_outputs = []
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            stage_outputs.append(x)

            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)

        return stage_outputs


class SimpleResBlock(nn.Module):
    """A minimal residual block for patch-grid feature aggregation."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        nn.init.constant_(self.norm2.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return shortcut + self.drop_path(x)


class SafeSpatialDownsample(nn.Module):
    """Downsample by ~2x while keeping each spatial dimension at least 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_h = max(1, x.shape[-2] // 2)
        target_w = max(1, x.shape[-1] // 2)
        return F.adaptive_avg_pool2d(x, (target_h, target_w))


class HighResResNetPyramid(nn.Module):
    """A simple ResNet-style pyramid with pooled stage transitions."""

    def __init__(self,
                 in_chans: int = 768,
                 depths: Iterable[int] = (1, 1, 1),
                 dims: Iterable[int] = (768, 1024, 1536),
                 drop_path_rate: float = 0.1,
                 num_downsamples: int = 2):
        super().__init__()

        depths = list(depths)
        dims = list(dims)
        self.dims = dims

        if len(depths) != len(dims):
            raise ValueError(f"depths and dims must have same length, got {len(depths)} and {len(dims)}")
        if num_downsamples > len(depths) - 1:
            raise ValueError(f"num_downsamples ({num_downsamples}) cannot exceed len(depths)-1 ({len(depths)-1})")

        self.input_proj = nn.Conv2d(in_chans, dims[0], kernel_size=1, bias=True) if in_chans != dims[0] else None

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = nn.ModuleList([SimpleResBlock(dim=dim, drop_path=dp_rates[cur + j]) for j in range(depth)])
            self.stages.append(blocks)
            cur += depth

            if i < num_downsamples:
                self.downsample_layers.append(
                    nn.Sequential(
                        SafeSpatialDownsample(),
                        nn.Conv2d(dim, dims[i + 1], kernel_size=1, bias=True),
                        LayerNorm(dims[i + 1], eps=1e-6, data_format='channels_first'),
                    ))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if self.input_proj is not None:
            x = self.input_proj(x)

        stage_outputs = []
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            stage_outputs.append(x)

            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)

        return stage_outputs


class HighResPoolingPyramid(nn.Module):
    """A parameter-free pooling backbone for patch ablations.

    It builds a true spatial pyramid using repeated pooling steps, so each
    returned stage corresponds to a different resolution without any learned
    convolution weights.
    """

    def __init__(self,
                 in_chans: int = 768,
                 dims: Iterable[int] = (768, 768, 768),
                 num_downsamples: int = 2,
                 pool_type: Literal['avg', 'max'] = 'avg'):
        super().__init__()
        self.in_chans = in_chans
        self.dims = list(dims)
        self.num_downsamples = int(num_downsamples)
        self.pool_type = pool_type
        if any(dim != in_chans for dim in self.dims):
            raise ValueError(
                f"HighResPoolingPyramid requires dims to equal in_chans ({in_chans}), "
                f"got dims={self.dims}"
            )
        if pool_type not in ('avg', 'max'):
            raise ValueError(f"Unsupported pool_type={pool_type}")

    def forward(self, x: torch.Tensor):
        if len(self.dims) == 1:
            cur = x
            for _ in range(self.num_downsamples):
                target_h = max(1, cur.shape[-2] // 2)
                target_w = max(1, cur.shape[-1] // 2)
                if self.pool_type == 'avg':
                    cur = F.adaptive_avg_pool2d(cur, (target_h, target_w))
                else:
                    cur = F.adaptive_max_pool2d(cur, (target_h, target_w))
            return [cur]

        stage_outputs = []
        cur = x
        num_stages = len(self.dims)
        num_identity_stages = max(0, num_stages - self.num_downsamples - 1)

        for stage_idx in range(num_stages):
            stage_outputs.append(cur)
            if stage_idx < num_stages - 1 and stage_idx >= num_identity_stages:
                target_h = max(1, cur.shape[-2] // 2)
                target_w = max(1, cur.shape[-1] // 2)
                if self.pool_type == 'avg':
                    cur = F.adaptive_avg_pool2d(cur, (target_h, target_w))
                else:
                    cur = F.adaptive_max_pool2d(cur, (target_h, target_w))

        return stage_outputs

class RotaryEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding (RoPE)
    """
    
    def __init__(self, dim, base=10000):
        """
        Args:
            dim (int): Rotary dimension, must be divisible by 4.
            base (int): Frequency base. Default: 10000.
        """
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")
        self.dim = dim
        self.base = base
        self.dim_per_axis = dim // 2
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, C, H, W)
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        B, C, H, W = x.shape
        if C < self.dim:
            raise ValueError(f"Input channels {C} must be >= rotary dim {self.dim}")
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.dim_per_axis, 2, device=x.device, dtype=torch.float32) 
                / self.dim_per_axis
            )
        )
        pos_h = torch.arange(H, device=x.device, dtype=torch.float32)
        pos_w = torch.arange(W, device=x.device, dtype=torch.float32)
        freqs_h = torch.outer(pos_h, inv_freq)
        freqs_w = torch.outer(pos_w, inv_freq)
        freqs_h_2d = freqs_h[:, None, :].expand(H, W, -1)
        freqs_w_2d = freqs_w[None, :, :].expand(H, W, -1)
        freqs_2d = torch.cat([freqs_h_2d, freqs_w_2d], dim=-1)
        freqs_2d = torch.cat([freqs_2d, freqs_2d], dim=-1)
        freqs_2d = freqs_2d.permute(2, 0, 1).unsqueeze(0)
        cos_emb = freqs_2d.cos()
        sin_emb = freqs_2d.sin()
        x_rot = x[:, :self.dim]
        x_pass = x[:, self.dim:]
        x_rot_pairs = x_rot.reshape(B, self.dim // 2, 2, H, W)
        x_real = x_rot_pairs[:, :, 0]
        x_imag = x_rot_pairs[:, :, 1]
        cos_vals = cos_emb[:, ::2]
        sin_vals = sin_emb[:, ::2]
        x_real_rot = x_real * cos_vals - x_imag * sin_vals
        x_imag_rot = x_real * sin_vals + x_imag * cos_vals
        x_rotated = torch.stack([x_real_rot, x_imag_rot], dim=2).reshape(B, self.dim, H, W)
        return torch.cat([x_rotated, x_pass], dim=1)
    
class PositionalEmbedding2DSinusoidal(nn.Module):
    """
    Adds 2D sinusoidal positional embeddings to a 4D tensor with scale balancing.
    The input tensor is expected to have the shape (B, C, H, W).
    
    Args:
        d_model (int): Model dimension, must be divisible by 4
        temperature (int): Temperature for sinusoidal encoding
        scale_mode (str): How to balance scales between input and positional embedding
            - 'learned': Use learnable scaling parameters (recommended)
            - 'normalize': Normalize both inputs to similar scales
            - 'adaptive': Adaptively scale based on input statistics
            - 'none': Direct addition (original behavior)
        init_pe_scale (float): Initial scale for positional embedding when using learned scaling
    """
    def __init__(self, d_model, temperature=10000, scale_mode='learned', init_pe_scale=0.1):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")
        
        self.d_model = d_model
        self.temperature = temperature
        self.scale_mode = scale_mode
        
        if scale_mode == 'learned':
            # Learnable scaling parameters
            self.input_scale = nn.Parameter(torch.ones(1))
            self.pe_scale = nn.Parameter(torch.full((1,), init_pe_scale))
        elif scale_mode == 'normalize':
            # Layer normalization for both inputs
            self.input_norm = nn.LayerNorm(d_model)
            self.pe_norm = nn.LayerNorm(d_model)
        elif scale_mode == 'adaptive':
            # Adaptive scaling based on input statistics
            self.momentum = 0.1
            self.register_buffer('running_input_std', torch.ones(1))
            self.register_buffer('running_pe_std', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W). C must equal d_model.
        
        Returns:
            torch.Tensor: Tensor with added positional embeddings, of the same shape.
        """
        B, C, H, W = x.shape
        if C != self.d_model:
            raise ValueError(f"Input channel {C} does not match d_model {self.d_model}")
        
        # Create coordinate grids
        y_pos = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(H, 1)
        
        # Normalize coordinates to [0, 1]
        y_pos = y_pos / H
        x_pos = x_pos / W
        
        # Calculate dimension indices
        dim_t = torch.arange(self.d_model // 4, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t / (self.d_model // 4))
        
        # Calculate positional embeddings
        pos_x = x_pos.unsqueeze(-1) / dim_t
        pos_y = y_pos.unsqueeze(-1) / dim_t
        
        # Apply sin/cos
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        
        # Concatenate x and y embeddings
        pos_emb = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, d_model)
        
        # Reshape to (1, d_model, H, W) for broadcasting
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0)
        
        # Apply different scale balancing strategies
        if self.scale_mode == 'learned':
            # Use learnable scaling parameters
            return self.input_scale * x + self.pe_scale * pos_emb
            
        elif self.scale_mode == 'normalize':
            # Normalize both inputs to similar scales
            x_flat = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            pe_flat = pos_emb.permute(0, 2, 3, 1)  # (1, H, W, C)
            
            x_norm = self.input_norm(x_flat)
            pe_norm = self.pe_norm(pe_flat)
            
            result = (x_norm + pe_norm).permute(0, 3, 1, 2)  # Back to (B, C, H, W)
            return result
            
        elif self.scale_mode == 'adaptive':
            # Adaptive scaling based on running statistics
            with torch.no_grad():
                input_std = x.std()
                pe_std = pos_emb.std()
                
                if self.training:
                    # Update running statistics during training
                    self.running_input_std.mul_(1 - self.momentum).add_(input_std * self.momentum)
                    self.running_pe_std.mul_(1 - self.momentum).add_(pe_std * self.momentum)
                
                # Use running statistics for scaling
                input_std_norm = self.running_input_std
                pe_std_norm = self.running_pe_std
            
            # Scale positional embedding to match input scale
            scale_factor = input_std_norm / (pe_std_norm + 1e-8)
            return x + scale_factor * pos_emb
            
        else:  # scale_mode == 'none'
            # Original direct addition
            return x + pos_emb


class DynamicSinusoidalPE2D(nn.Module):
    """Dynamic 2D sinusoidal positional encoding for patch feature grids."""

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")
        self.d_model = d_model
        self.temperature = temperature

    def forward(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        half = self.d_model // 2
        freq = torch.arange(0, half, 2, device=device, dtype=torch.float32) / half
        freq = 1.0 / (self.temperature ** freq)

        pos_h = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)
        pos_w = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1)

        enc_h = torch.cat([torch.sin(pos_h * freq), torch.cos(pos_h * freq)], dim=1)
        enc_w = torch.cat([torch.sin(pos_w * freq), torch.cos(pos_w * freq)], dim=1)

        pe = torch.zeros(self.d_model, height, width, device=device, dtype=torch.float32)
        pe[:half] = enc_h.t().unsqueeze(2).expand(-1, -1, width)
        pe[half:] = enc_w.t().unsqueeze(1).expand(-1, height, -1)
        return pe.unsqueeze(0).to(dtype=dtype)


class PatchAdapter(nn.Module):
    """Lightweight local patch adapter without downsampling.

    This module only injects local context and 2D spatial position information
    while keeping the patch grid resolution and channel count unchanged.
    Token-count reduction is handled later by `WindowRouter`.
    """

    def __init__(
        self,
        d_patch: int,
        num_blocks: int = 2,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.pe = DynamicSinusoidalPE2D(d_patch)
        drop_rates = torch.linspace(0, drop_path_rate, steps=max(num_blocks, 1)).tolist()
        self.blocks = nn.Sequential(*[
            ConvNeXtV2Block(dim=d_patch, drop_path=drop_rates[i])
            for i in range(num_blocks)
        ])

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        _, _, h, w = patch_feats.shape
        pos = self.pe(h, w, patch_feats.device, patch_feats.dtype)
        return self.blocks(patch_feats + pos)


class WindowScorer(nn.Module):
    """Patch-window scoring head with local-global fusion."""

    def __init__(self, d_patch: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.GroupNorm(1, 2 * d_patch, eps=1e-6, affine=True)
        self.scorer = nn.Sequential(
            nn.Conv2d(2 * d_patch, d_patch, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(d_patch, 1, kernel_size=1, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                gain = 0.05 if module.out_channels == 1 else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, adapted_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global_ctx = self.global_pool(adapted_feats).expand_as(adapted_feats)
        fused = self.norm(torch.cat([adapted_feats, global_ctx], dim=1))
        score_logits = self.scorer(fused)
        score_map = torch.sigmoid(score_logits)
        return score_logits, score_map


class LLMProjector(nn.Module):
    """Project routed visual tokens to the LLM hidden size.

    Token-count reduction is handled upstream by `WindowRouter`, so this module
    only remaps each routed token from patch space into the LLM hidden space.
    """

    num_position_rows = 4

    def __init__(self, d_patch: int, h_llm: int):
        super().__init__()
        hidden_dim = 2 * d_patch
        self.shortcut = nn.Linear(d_patch, h_llm)
        self.mlp = nn.Sequential(
            nn.Linear(d_patch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, h_llm),
        )
        self.residual_gate = nn.Parameter(torch.tensor(-4.0))
        self.norm = nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.shortcut.weight, gain=0.1)
        nn.init.zeros_(self.shortcut.bias)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                gain = 0.05 if module.out_features == self.shortcut.out_features else 0.5
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, routed_tokens: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(routed_tokens)
        residual = self.mlp(routed_tokens)
        gate = torch.sigmoid(self.residual_gate)
        return self.norm(shortcut + gate * residual)


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
        output_gate_init: float = 1.0,
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
        self.output_gate_init = float(output_gate_init)
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
        attn_kwargs = dict(embed_dim=self.resampler_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.prompt_attn = nn.MultiheadAttention(**attn_kwargs)
        self.decoder_blocks = nn.ModuleList([
            PatchDecoderBlock(self.resampler_dim, num_heads, dropout)
            for _ in range(self.num_layers)
        ])
        self.to_llm = nn.Linear(self.resampler_dim, self.llm_hidden_size)
        self.output_norm = nn.LayerNorm(self.llm_hidden_size)
        self.output_gate = nn.Parameter(torch.tensor(self.output_gate_init, dtype=torch.float32))
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

    def _condition_queries(self, prompt_tokens, prompt_mask):
        batch_size = prompt_tokens.size(0)
        queries = (self.query_tokens + self.query_id).unsqueeze(0).expand(batch_size, -1, -1)
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
        visual_tokens = self._condition_queries(prompt_tokens, prompt_attention_mask)
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

        patch_attention = patch_attn_heads.mean(dim=1).float()
        patch_attention = patch_attention.masked_fill(~valid_mask.flatten(1).unsqueeze(1), 0.0)
        patch_attention = patch_attention / patch_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        coord_indices = self._coord_indices(b, h, w, features.device).float()
        token_positions = torch.bmm(patch_attention, coord_indices).round().long()
        token_positions[..., 0].clamp_(0, h - 1)
        token_positions[..., 1].clamp_(0, w - 1)
        token_valid = torch.ones((b, self.num_query), dtype=torch.bool, device=features.device)
        llm_tokens = self.output_norm(self.to_llm(visual_tokens))
        llm_tokens = llm_tokens * self.output_gate.to(device=llm_tokens.device, dtype=llm_tokens.dtype)

        return {
            'visual_tokens': llm_tokens,
            'token_positions': token_positions,
            'token_valid': token_valid,
            'patch_attention': patch_attention.view(b, self.num_query, h, w),
            'patch_valid_mask': valid_mask,
            'region_attention': patch_attention,
            'region_attention_heads': patch_attn_heads,
            'visual_to_region_attention': None,
            'visual_to_region_attention_heads': None,
        }


class WindowRouter(nn.Module):
    """Dynamic window router with hard top-k forward and STE backward."""

    def __init__(
        self,
        d_patch: int,
        window_size: int = 4,
        topk_windows: int = 64,
        alpha: float = 0.5,
        tau_ste: float = 0.5,
        tau_pool: float = 1.0,
        tau_aux: float = 0.7,
        window_score_topn: int = 2,
    ):
        super().__init__()
        self.d_patch = d_patch
        self.window_size = window_size
        self.topk_windows = topk_windows
        self.alpha = alpha
        self.tau_ste = tau_ste
        self.tau_pool = tau_pool
        self.tau_aux = tau_aux
        self.window_score_topn = window_score_topn

    @staticmethod
    def _pad_to_window(tensor: torch.Tensor, window_size: int, pad_value: float) -> torch.Tensor:
        _, _, h, w = tensor.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=pad_value)
        return tensor

    @staticmethod
    def _partition_windows(tensor: torch.Tensor, ws: int) -> torch.Tensor:
        b, c, h, w = tensor.shape
        n_h, n_w = h // ws, w // ws
        x = tensor.reshape(b, c, n_h, ws, n_w, ws)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, n_h * n_w, ws * ws, c)

    @staticmethod
    def _generate_coords(h: int, w: int, ws: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        coords = torch.stack([grid_r, grid_c], dim=-1)
        n_h, n_w = h // ws, w // ws
        coords = coords.reshape(n_h, ws, n_w, ws, 2)
        coords = coords.permute(0, 2, 1, 3, 4)
        return coords.reshape(1, n_h * n_w, ws * ws, 2)

    def _compute_window_scores(
        self,
        window_logits: torch.Tensor,
        window_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = window_logits.float().masked_fill(~window_valid, -1e4)
        top_n = min(self.window_score_topn, logits.size(-1))
        topn_logits, _ = logits.topk(top_n, dim=2)
        window_score = topn_logits.mean(dim=2)
        window_is_valid = window_valid.any(dim=2)
        neg_inf = torch.full_like(window_score, -1e4)
        window_score = torch.where(window_is_valid, window_score, neg_inf)

        valid_count = window_is_valid.sum(dim=1, keepdim=True).clamp(min=1)
        score_mean = window_score.masked_fill(~window_is_valid, 0.0).sum(dim=1, keepdim=True) / valid_count
        score_var = (
            (window_score - score_mean)
            .masked_fill(~window_is_valid, 0.0)
            .pow(2)
            .sum(dim=1, keepdim=True) / valid_count
        )
        score_std = torch.sqrt(score_var + 1e-6).clamp_min(1e-6)
        window_score = torch.where(window_is_valid, (window_score - score_mean) / score_std, neg_inf)
        return window_score, window_is_valid

    def _topk_with_ste(
        self,
        window_score: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n = window_score.shape
        k = min(self.topk_windows, n)
        safe_scores = torch.where(valid_mask, window_score, torch.full_like(window_score, -1e9))
        topk_vals, topk_idx = torch.topk(safe_scores, k=k, dim=1)

        valid_count = valid_mask.sum(dim=1)
        actual_k = torch.clamp(valid_count, max=k)

        threshold_index = torch.clamp(actual_k - 1, min=0).unsqueeze(1)
        threshold = topk_vals.gather(1, threshold_index).detach()

        rank_mask = (
            torch.arange(k, device=window_score.device).unsqueeze(0) < actual_k.unsqueeze(1)
        )
        hard_mask = torch.zeros_like(window_score, dtype=torch.float32)
        hard_mask.scatter_(1, topk_idx, rank_mask.to(dtype=hard_mask.dtype))
        hard_mask = hard_mask * valid_mask.to(dtype=hard_mask.dtype)

        soft_mask = torch.sigmoid((safe_scores - threshold) / self.tau_ste)
        soft_mask = soft_mask * valid_mask.to(dtype=soft_mask.dtype)
        mask = soft_mask + (hard_mask - soft_mask).detach()

        return mask, hard_mask.bool(), topk_idx, actual_k

    @staticmethod
    def _gather_tensor(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if src.dim() == 2:
            return torch.gather(src, 1, idx)
        expand_shape = [idx.size(0), idx.size(1)] + list(src.shape[2:])
        idx_expanded = idx.view(idx.size(0), idx.size(1), *([1] * (src.dim() - 2))).expand(*expand_shape)
        return torch.gather(src, 1, idx_expanded)

    def _low_window_indices(
        self,
        window_score: torch.Tensor,
        selected_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = window_score.size(1)
        low_count = max(n - min(self.topk_windows, n), 0)
        if low_count == 0:
            return window_score.new_zeros((window_score.size(0), 0), dtype=torch.long)

        big = torch.full_like(window_score, 1e6)
        priority = torch.where(
            ~valid_mask,
            big + 1e6,
            torch.where(selected_mask, big, -window_score),
        )
        return torch.argsort(priority, dim=1)[:, :low_count]

    def _collect_aux_materials(
        self,
        score_logits: torch.Tensor,
        adapted_feats: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, d, h, w = adapted_feats.shape
        feats_flat = adapted_feats.reshape(b, d, h * w).transpose(1, 2)
        logits_flat = score_logits.reshape(b, h * w).float() / self.tau_aux
        valid_flat = valid_mask.reshape(b, h * w)
        logits_flat = logits_flat.masked_fill(~valid_flat, -1e4)
        attn = torch.softmax(logits_flat, dim=1)
        attn = attn * valid_flat.to(dtype=attn.dtype)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        return (attn.unsqueeze(-1) * feats_flat).sum(dim=1)

    def forward(
        self,
        adapted_feats: torch.Tensor,
        score_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, d, h, w = adapted_feats.shape
        ws = self.window_size

        if valid_mask is None:
            valid_mask = torch.ones((b, 1, h, w), device=adapted_feats.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        feats_pad = self._pad_to_window(adapted_feats, ws, pad_value=0.0)
        logits_pad = self._pad_to_window(score_logits, ws, pad_value=-1e9)
        vmask_pad = self._pad_to_window(valid_mask.to(dtype=adapted_feats.dtype), ws, pad_value=0.0).bool()

        h_pad, w_pad = feats_pad.shape[-2:]
        window_feats = self._partition_windows(feats_pad, ws)
        window_logits = self._partition_windows(logits_pad, ws).squeeze(-1)
        window_valid = self._partition_windows(vmask_pad.to(dtype=feats_pad.dtype), ws).squeeze(-1).bool()
        window_coords = self._generate_coords(h_pad, w_pad, ws, adapted_feats.device).expand(b, -1, -1, -1)

        window_score, window_is_valid = self._compute_window_scores(window_logits, window_valid)
        ste_mask, hard_mask, topk_idx, _ = self._topk_with_ste(window_score, window_is_valid)

        high_feats = self._gather_tensor(window_feats, topk_idx)
        high_logits = self._gather_tensor(window_logits, topk_idx)
        high_valid = self._gather_tensor(window_valid, topk_idx)
        high_coords = self._gather_tensor(window_coords, topk_idx)
        high_selected = self._gather_tensor(hard_mask, topk_idx)
        high_valid = high_valid & high_selected.unsqueeze(-1)

        high_scores = torch.sigmoid(high_logits.clamp(min=-8.0, max=8.0))
        high_feats = high_feats * (1.0 + self.alpha * high_scores.unsqueeze(-1))
        high_tokens = high_feats.reshape(b, -1, d)
        high_coords = high_coords.reshape(b, -1, 2)
        high_token_valid = high_valid.reshape(b, -1)
        high_tokens = high_tokens * high_token_valid.unsqueeze(-1).to(dtype=high_tokens.dtype)

        low_idx = self._low_window_indices(window_score, hard_mask, window_is_valid)
        low_feats = self._gather_tensor(window_feats, low_idx)
        low_logits = self._gather_tensor(window_logits, low_idx)
        low_valid = self._gather_tensor(window_valid, low_idx)
        low_coords = self._gather_tensor(window_coords, low_idx)
        low_selected = self._gather_tensor(hard_mask, low_idx)

        if low_feats.numel() > 0:
            logits_for_pool = (low_logits.float() / self.tau_pool).masked_fill(~low_valid, -1e4)
            attn = torch.softmax(logits_for_pool, dim=2)
            attn = attn * low_valid.to(dtype=attn.dtype)
            attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            low_summary = (attn.unsqueeze(-1) * low_feats).sum(dim=2)
            low_center = low_coords.float().mean(dim=2).round().long()
            low_window_valid = low_valid.any(dim=2) & ~low_selected
            low_summary = low_summary * low_window_valid.unsqueeze(-1).to(dtype=low_summary.dtype)
        else:
            low_summary = adapted_feats.new_zeros((b, 0, d))
            low_center = torch.zeros((b, 0, 2), device=adapted_feats.device, dtype=torch.long)
            low_window_valid = torch.zeros((b, 0), device=adapted_feats.device, dtype=torch.bool)

        routed_tokens = torch.cat([high_tokens, low_summary], dim=1)
        token_positions = torch.cat([high_coords, low_center], dim=1)
        token_valid = torch.cat([high_token_valid, low_window_valid], dim=1)
        routed_tokens = routed_tokens * token_valid.unsqueeze(-1).to(dtype=routed_tokens.dtype)

        slide_repr = self._collect_aux_materials(score_logits, adapted_feats, valid_mask)

        return {
            'routed_tokens': routed_tokens,
            'token_positions': token_positions,
            'token_valid': token_valid,
            'slide_repr': slide_repr,
            'window_scores': window_score,
            'ste_mask': ste_mask,
            'selected_window_mask': hard_mask,
        }


class CustomQformer(nn.Module):
    """
    A custom, flexible Q-Former module designed to be trained from scratch.

    This implementation faithfully replicates the logic of the original InstructBLIP
    Q-Former by using its core components. It allows for full control over the
    input embeddings, enabling the use of an external word embedding layer (like Qwen3's)
    via a projection layer. It correctly applies position embeddings, LayerNorm, and
    Dropout before feeding the data into the main transformer encoder.

    Args:
        config (InstructBlipQFormerConfig):
            Configuration for the Q-Former, defining its internal architecture.
        word_embeddings (nn.Embedding):
            The pretrained word embedding layer from the target language model.
    """
    def __init__(
        self,
        config: InstructBlipQFormerConfig,
        word_embeddings: nn.Embedding,
    ):
        super().__init__()
        self.config = config

        # --- Core Components ---
        
        # 1. External word embeddings from the main LLM
        self.word_embeddings = word_embeddings
        embedding_dim = self.word_embeddings.embedding_dim
        qformer_hidden_size = config.hidden_size
        
        # 2. Projection layer to adapt LLM embeddings to the Q-Former's hidden size
        self.text_input_projection = nn.Linear(embedding_dim, qformer_hidden_size)

        # 3. The main stack of transformer layers
        self.encoder = InstructBlipQFormerEncoder(config)

        # 4. Learnable query tokens, which act as the interface to the LLM
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, qformer_hidden_size) * 0.02
        )

        # 5. Position embeddings - only create if using absolute position embedding
        if config.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, qformer_hidden_size
            )
        else:
            # For relative position embedding, the encoder will handle position encoding internally
            self.position_embeddings = None
        
        # 6. LayerNorm and Dropout, applied after combining query and text embeddings
        # This is a critical step replicated from the original implementation.
        self.layernorm = nn.LayerNorm(qformer_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that fuses visual and textual information.

        Args:
            input_ids (torch.LongTensor): Token IDs for the text prompt.
                Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask for the text prompt.
                Shape: (batch_size, seq_len)
            encoder_hidden_states (torch.Tensor): Output features from the vision encoder.
                Shape: (batch_size, num_patches, image_feature_dim)
            encoder_attention_mask (Optional[torch.Tensor]): Mask for image features.

        Returns:
            torch.Tensor: The fused query embeddings.
                Shape: (batch_size, num_query_tokens, qformer_hidden_size)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        num_queries = self.query_tokens.shape[1]

        # 1. Prepare text embeddings: embed, project, and conditionally add position encodings.
        
        text_word_embeds = self.word_embeddings(input_ids)
        projected_text_embeds = self.text_input_projection(text_word_embeds)
        
        # Only add absolute position embeddings if they exist
        if self.position_embeddings is not None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).expand(batch_size, -1)
            text_pos_embeds = self.position_embeddings(position_ids)
            final_text_embeds = projected_text_embeds + text_pos_embeds
        else:
            # For relative position embedding, don't add position encodings here
            final_text_embeds = projected_text_embeds

        # 2. Prepare query embeddings for the batch.
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)

        # 3. Concatenate query and text embeddings and apply LayerNorm + Dropout.
        # This mirrors the behavior of the original `InstructBlipQFormerEmbeddings`.
        embedding_output = torch.cat([query_embeds, final_text_embeds], dim=1)
        embedding_output = self.layernorm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        # 4. Create attention masks for the combined sequence and the encoder.
        query_attention_mask = torch.ones((batch_size, num_queries), dtype=torch.long, device=device)
        combined_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)
        extended_attention_mask = self.get_extended_attention_mask(combined_attention_mask)
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2], dtype=torch.long, device=device
            )
        extended_encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        # 5. Pass inputs to the core encoder.
        # The `query_length` argument is crucial for the internal cross-attention.
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=extended_encoder_attention_mask,
            query_length=num_queries,
            return_dict=True,
        )

        # 6. Extract the hidden states corresponding to the query tokens.
        last_hidden_state = encoder_outputs.last_hidden_state
        query_output = last_hidden_state[:, :num_queries, :]

        return query_output

    # Helper functions to create broadcastable attention masks.
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        else:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float32)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        return encoder_extended_attention_mask
    

EPS = 1e-8  # Small epsilon to avoid division by zero


class AttentionPooling(nn.Module):
    """
    Cross-attention style pooling for both regression and survival prediction.
    Query: (N, q_dim)  from special token hidden state (<REG> or <SRV>)
    Key/Value: visual feature map (N, C, H, W)
    Optional binary mask: (N, 1, H, W), 1 for valid locations.
    Output: (N, hidden_dim)
    """
    def __init__(self, q_dim: int, kv_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Enable bias for better gradient flow
        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=True)
        self.k_proj = nn.Conv2d(kv_dim, hidden_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(kv_dim, hidden_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Better initialization
        self._init_weights()

    
    def _init_weights(self):
        """Initialize weights for stable cross-attention training."""
        # Xavier initialization for linear layers
        for module in [self.q_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        
        # Xavier initialization for conv layers
        for module in [self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q: (N, q_dim)
        # kv: (N, C, H, W)
        N, C, H, W = kv.shape
        
        # Project query, key, and value
        q_proj = self.q_proj(q)                                   # (N, Hdim)
        k = self.k_proj(kv).flatten(2).transpose(1, 2)            # (N, HW, Hdim)
        v = self.v_proj(kv).flatten(2).transpose(1, 2)            # (N, HW, Hdim)

        # Compute attention with improved numerical stability
        q_proj = q_proj.unsqueeze(1)                              # (N, 1, Hdim)
        attn_scores = torch.matmul(q_proj, k.transpose(-2, -1))   # (N, 1, HW)
        
        # Use temperature scaling for better numerical stability
        temperature = max(1.0, (self.hidden_dim ** 0.5))
        attn_scores = attn_scores / temperature

        if mask is not None:
            # Ensure mask is properly shaped and processed
            if mask.dim() == 4:  # (N, 1, H, W)
                mask_flat = (mask > 0).float().flatten(2).squeeze(1)  # (N, HW)
            elif mask.dim() == 3:  # (N, H, W)
                mask_flat = (mask > 0).float().flatten(1)  # (N, HW)
            else:
                mask_flat = mask  # Assume already flattened
            
            # Apply mask with improved numerical stability
            attn_scores = attn_scores.masked_fill(
                mask_flat.unsqueeze(1) == 0, -1e9  # Use -1e9 instead of -inf for better stability
            )

        # Apply softmax with clamping for numerical stability
        attn_scores = torch.clamp(attn_scores, min=-10, max=10)
        attn = torch.softmax(attn_scores, dim=-1)                 # (N, 1, HW)
        
        # Apply attention weights to values
        context = torch.matmul(attn, v)                           # (N, 1, Hdim)
        context = context.squeeze(1)                              # (N, Hdim)
        
        # Apply output projection
        output = self.out_proj(context)
        
        return output


EPS = 1e-8  # Small epsilon to avoid division by zero


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


# ========== WSI Feature Projector ==========

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
        
        # Learnable scale factors for combining projections (optional enhancement)
        self.scale_factors = nn.Parameter(torch.ones(len(wsi_input_dims)))
        
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
        for i, (feat, projector, norm, scale) in enumerate(
                zip(wsi_features, self.projectors, self.post_norms, self.scale_factors)):
            # Validate input dimension
            if feat.size(-1) != self.wsi_input_dims[i]:
                raise ValueError(
                    f"WSI source {i}: expected dim {self.wsi_input_dims[i]}, got {feat.size(-1)}"
                )
            
            # Project
            proj = projector(feat)  # (B, llm_hidden_size)
            proj = norm(proj)
            
            # Apply scale factor
            proj = proj * scale
            
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
        
        return self.dropout(self.projectors[source_idx](wsi_feature) * self.scale_factors[source_idx])
    
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
