import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Conv2d):
    """
    Partial Convolution layer, as described in "Image Inpainting for Irregular Holes Using Partial Convolutions".
    This version includes caching for efficiency and correct slide_winsize calculation.
    """
    def __init__(self, *args, **kwargs):
        # Pop custom arguments before passing to parent
        self.multi_channel = kwargs.pop('multi_channel', False)
        self.return_mask = kwargs.pop('return_mask', False)
        super(PartialConv2d, self).__init__(*args, **kwargs)

        # Create mask update kernel
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        
        # Correctly calculate sliding window size based on convolution type.
        # For standard conv (groups=1), window size is C_in * K_h * K_w.
        # For depthwise conv (groups=C_in), each filter sees 1 channel, so it's K_h * K_w.
        self.slide_winsize = (self.in_channels / self.groups) * self.kernel_size[0] * self.kernel_size[1]
        
        # Caching for performance
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        
        # Initialize mask if not provided
        if mask_in is None:
            if self.multi_channel:
                mask = torch.ones_like(input)
            else:
                mask = torch.ones(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device, dtype=input.dtype)
        else:
            mask = mask_in
            
        # Only update mask tensors if input size changes.
        if self.update_mask is None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            
            with torch.no_grad():
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = self.mask_ratio * self.update_mask

        # Apply mask to input and perform convolution
        masked_input = input * mask
        raw_out = super(PartialConv2d, self).forward(masked_input)

        # Apply normalization and handle bias
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (raw_out - bias_view) * self.mask_ratio + bias_view * self.update_mask
        else:
            output = raw_out * self.mask_ratio

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class LayerNorm2d(nn.LayerNorm):
    """
    Channel-wise LayerNorm for 4D tensors (B, C, H, W).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x

class DropPath(nn.Module):
    """
    Stochastic depth implementation.
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class PartialConvMlp(nn.Module):
    """
    MLP Block using 1x1 PartialConvs.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = PartialConv2d(in_features, hidden_features, kernel_size=1, return_mask=True)
        self.act = act_layer()
        self.fc2 = PartialConv2d(hidden_features, out_features, kernel_size=1, return_mask=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        x, mask = self.fc1(x, mask)
        x = self.act(x)
        x = self.drop(x)
        x, mask = self.fc2(x, mask)
        x = self.drop(x)
        return x, mask

class PartialConvNeXtBlock(nn.Module):
    """ 
    ConvNeXt Block adapted for Partial Convolution.
    """
    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, kernel_size=7, mlp_ratio=4):
        super().__init__()
        self.conv_dw = PartialConv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, return_mask=True)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.mlp = PartialConvMlp(in_features=dim, hidden_features=int(mlp_ratio * dim))
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask):
        shortcut = x
        
        # Main path
        x_main, mask_updated = self.conv_dw(x, mask)
        x_main = self.norm(x_main)
        x_main, mask_updated = self.mlp(x_main, mask_updated)
        
        if self.gamma is not None:
            x_main = x_main.mul(self.gamma.view(1, -1, 1, 1))

        # The residual connection must be constrained by the *updated* mask
        # to ensure the output feature map and the output mask are consistent.
        x_main_w_drop = self.drop_path(x_main)
        output = (shortcut * mask_updated) + x_main_w_drop
        
        return output, mask_updated

class PartialConvDownsample(nn.Module):
    """
    Downsampling layer: Conv -> Norm.
    It normalizes the features *after* their dimensions have been changed by the convolution.
    """
    def __init__(self, in_chs, out_chs, stride=2):
        super().__init__()
        self.conv = PartialConv2d(in_chs, out_chs, kernel_size=stride, stride=stride, return_mask=True)
        self.norm = LayerNorm2d(out_chs) # Initialize with OUTPUT channels

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        x = self.norm(x)
        return x, mask
    
class PartialConvNeXtStage(nn.Module):
    """
    A PartialConvNeXt Stage with multiple blocks.
    """
    def __init__(self, dim, depth, kernel_size=7, mlp_ratio=4., drop_path=0., ls_init_value=1e-6):
        super().__init__()
        
        if isinstance(drop_path, (list, tuple)):
            drop_path_rates = drop_path
        else:
            drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.ModuleList([
            PartialConvNeXtBlock(
                dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                drop_path=drop_path_rates[i], ls_init_value=ls_init_value
            ) for i in range(depth)
        ])

    def forward(self, x, mask):
        for block in self.blocks:
            x, mask = block(x, mask)
        return x, mask
    
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
    

class HighResPartialConvNeXt(nn.Module):
    """
    An example model tailored for high-resolution, high-dimensional feature map inputs.
    It features a "gentle" start, lightweight high-res stages, and heavier low-res stages.
    """
    def __init__(self, in_chans=768, 
                 depths=[2, 2, 6], 
                 dims=[768, 768, 768],
                 mlp_ratios=[2, 4, 4], # Use different mlp_ratios for different stages
                 kernel_sizes=[3, 7, 7], # Use different kernel_sizes for different stages
                 drop_path=0.1 # Add drop_path argument with default 0.
                ):
        super().__init__()
        
        # 1. Input Projection: No downsampling. Just a 1x1 conv to start the process.
        # This layer is optional if your input channel is already what you want.
        # self.input_projection = PartialConv2d(in_chans, dims[0], kernel_size=1, return_mask=True)

        # 2. Stage 0 (Full Resolution): Lightweight stage on the full HxW input.
        # Uses small kernel size and small mlp_ratio to save computation.
        print(f"Stage 0 (Full Res): dim={dims[0]}, depth={depths[0]}, k_size={kernel_sizes[0]}, mlp_ratio={mlp_ratios[0]}, drop_path={drop_path}")
        self.stage0 = PartialConvNeXtStage(dim=dims[0], depth=depths[0], 
                                           kernel_size=kernel_sizes[0], mlp_ratio=mlp_ratios[0], drop_path=drop_path)
        
        # 3. Downsample 1 -> Stage 1 (H/2, W/2)
        self.downsample1 = PartialConvDownsample(in_chs=dims[0], out_chs=dims[1], stride=2)
        print(f"Stage 1 (H/2, W/2): dim={dims[1]}, depth={depths[1]}, k_size={kernel_sizes[1]}, mlp_ratio={mlp_ratios[1]}, drop_path={drop_path}")
        self.stage1 = PartialConvNeXtStage(dim=dims[1], depth=depths[1],
                                           kernel_size=kernel_sizes[1], mlp_ratio=mlp_ratios[1], drop_path=drop_path)
        
        # 4. Downsample 2 -> Stage 2 (H/4, W/4): Computation is cheaper, can be heavier.
        self.downsample2 = PartialConvDownsample(in_chs=dims[1], out_chs=dims[2], stride=2)
        print(f"Stage 2 (H/4, W/4): dim={dims[2]}, depth={depths[2]}, k_size={kernel_sizes[2]}, mlp_ratio={mlp_ratios[2]}, drop_path={drop_path}")
        self.stage2 = PartialConvNeXtStage(dim=dims[2], depth=depths[2],
                                           kernel_size=kernel_sizes[2], mlp_ratio=mlp_ratios[2], drop_path=drop_path)
    def forward(self, x, mask):
        # Full resolution processing
        # x, mask = self.input_projection(x, mask)
        x, mask = self.stage0(x, mask)
        
        # Downsample and process
        x, mask = self.downsample1(x, mask)
        x, mask = self.stage1(x, mask)

        # Downsample and process again
        x, mask = self.downsample2(x, mask)
        x, mask = self.stage2(x, mask)
        
        return x, mask