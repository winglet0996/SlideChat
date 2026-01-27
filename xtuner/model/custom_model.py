import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from transformers import InstructBlipQFormerConfig
from transformers.models.instructblip.modeling_instructblip import InstructBlipQFormerEncoder
from typing import Optional, Tuple, Iterable, Literal

class PartialConv2d(nn.Conv2d):
    """
    Partial Convolution layer, as described in "Image Inpainting for Irregular Holes Using Partial Convolutions".
    This version is adapted to return the updated mask, which is essential for propagation.
    """
    def __init__(self, *args, **kwargs):
        # Whether the mask is multi-channel or not
        self.multi_channel = kwargs.pop('multi_channel', False)
        # Whether to return the mask
        self.return_mask = kwargs.pop('return_mask', True)
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            self.slide_winsize = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        else:
            weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        
        self.register_buffer('updater_buf', weight_maskUpdater) # Use a buffer

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        
        if mask_in is None:
            # if mask is not provided, create a ones mask
            if self.multi_channel:
                mask = torch.ones_like(input)
            else:
                mask = torch.ones(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device, dtype=input.dtype)
        else:
            mask = mask_in

        with torch.no_grad():
            # The updater does not require gradients
            update_mask = F.conv2d(mask, self.updater_buf, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            
            # For mixed precision training, ensure consistent dtypes
            mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            mask_ratio = mask_ratio.to(input.dtype)
            
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask

        # Apply the mask to the input
        masked_input = input * mask
        
        # Perform the convolution
        raw_out = super(PartialConv2d, self).forward(masked_input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (raw_out - bias_view) * mask_ratio + bias_view
        else:
            output = raw_out * mask_ratio

        if self.return_mask:
            return output, update_mask
        else:
            return output

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

class PartialConvNeXtV2Block(nn.Module):
    """ Partial ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # Use PartialConv2d for the depthwise convolution
        self.dwconv = PartialConv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask):
        shortcut = x
        # The dwconv is a PartialConv2d, it returns the feature map and the updated mask
        x, updated_mask = self.dwconv(x, mask)
        
        # Permute to (N, H, W, C) to use nn.Linear and the official GRN
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # Permute back to (N, C, H, W)

        # Apply residual connection.
        # The output of the main path is added to the original input.
        x = shortcut + self.drop_path(x)
        
        return x, updated_mask


class HighResPartialConvNeXt(nn.Module):
    """
    A PartialConvNeXtV2-based model optimized for processing high-dimensional feature maps.
    It removes the aggressive stem and classification head, acting as a general-purpose
    feature refinement module.

    Args:
        in_chans (int): Number of input feature channels.
        depths (tuple(int)): Number of blocks at each stage.
        dims (int): Feature dimension at each stage.
        drop_path_rate (float): Stochastic depth rate.
        num_downsamples (int): Number of downsampling stages. Must be <= len(depths) - 1.
    """
    def __init__(self, in_chans=768, 
                 depths=[2, 2, 6], dims=[768, 768, 768], 
                 drop_path_rate=0.1, num_downsamples=2
                 ):
        super().__init__()
        
        # Store dims for later access
        self.dims = dims
        
        if num_downsamples > len(depths) - 1:
            raise ValueError(f"num_downsamples ({num_downsamples}) cannot exceed len(depths)-1 ({len(depths)-1})")

        # --- Gentle Input Projection (1x1 Conv) ---
        # Only used if the input channels don't match the first stage dimension.
        if in_chans != dims[0]:
            self.input_proj = PartialConv2d(in_chans, dims[0], kernel_size=1)
        else:
            self.input_proj = None

        # --- Stages & Downsampling Layers ---
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Create all stages
        for i in range(len(depths)):
            stage = nn.ModuleList(
                [PartialConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

            # Create corresponding downsampling layer if needed
            if i < num_downsamples:
                downsample_layer = nn.ModuleList([
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    PartialConv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                ])
                self.downsample_layers.append(downsample_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (PartialConv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        """
        Input:
            x (torch.Tensor): Input feature map of shape (N, C_in, H, W).
            mask (torch.Tensor, optional): Input mask of shape (N, 1, H, W). Defaults to all ones.
        
        Returns:
            (List[torch.Tensor], torch.Tensor): A list of stage outputs and the final mask.
        """
        if mask is None:
            mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)

        if self.input_proj is not None:
            x, mask = self.input_proj(x, mask)

        stage_outputs = []
        # Iterate through stages and downsampling layers
        for i, stage in enumerate(self.stages):
            for block in stage:
                x, mask = block(x, mask)
            
            stage_outputs.append(x)

            if i < len(self.downsample_layers):
                # Apply downsampling
                x = self.downsample_layers[i][0](x) # LayerNorm
                x, mask = self.downsample_layers[i][1](x, mask) # PartialConv

        return stage_outputs, mask
    
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
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.head = nn.Linear(in_dim, out_dim)
            
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
        Forward pass.
        
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
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Cox proportional hazards partial likelihood loss.
    
    Args:
        theta: (N,) risk score (log-risk)
        time:  (N,) follow-up time
        event: (N,) 1=event, 0=censored
    """
    theta = theta.view(-1)
    time = time.view(-1)
    event = event.view(-1).float()

    n_events = event.sum()
    if n_events < 1:
        return torch.zeros((), device=theta.device, dtype=theta.dtype)

    # Risk set mask: R[i,j] = (time[j] >= time[i])
    R = (time[None, :] >= time[:, None])

    # log denom_i = log sum_{j in R_i} exp(theta_j)
    M = theta.numel()
    theta_row = theta.view(1, M).expand(M, M)
    log_denom = torch.logsumexp(theta_row.masked_fill(~R, float("-inf")), dim=1)

    # negative partial log-likelihood
    loss = -((theta - log_denom) * event).sum() / (n_events + eps)
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
        for i, (feat, projector, scale) in enumerate(zip(wsi_features, self.projectors, self.scale_factors)):
            # Validate input dimension
            if feat.size(-1) != self.wsi_input_dims[i]:
                raise ValueError(
                    f"WSI source {i}: expected dim {self.wsi_input_dims[i]}, got {feat.size(-1)}"
                )
            
            # Project
            proj = projector(feat)  # (B, llm_hidden_size)
            
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