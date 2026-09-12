# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import numpy as np
import requests
from PIL import Image
import h5py
import torch
from torchvision import transforms

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


def get_bos_eos_token_ids(tokenizer):
    if tokenizer.__class__.__name__ in [
            'QWenTokenizer', 'QWen2Tokenizer', 'Qwen2Tokenizer',
            'Qwen2TokenizerFast'
    ]:
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, \
            'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        bos_token_id = []
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    return bos_token_id, eos_token_id


def encode_fn(example,
              tokenizer,
              max_length,
              max_patch_num,
              input_ids_with_output=True,
              with_image_token=False,
              per_image_length=None):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    n_images = 0
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(DEFAULT_IMAGE_TOKEN)
            ]
            # assert len(chunk_encode) == 2
            n_images += len(chunk_encode) - 1
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if max_patch_num is not None and max_patch_num > 0:
        per_image_length = example.get('image_len', per_image_length)
        per_image_length = min(per_image_length, max_patch_num)
        input_ids = input_ids[:max_length - n_images * per_image_length]
        labels = labels[:max_length - n_images * per_image_length]
    else:
        per_image_length = max_length//2 # hard code for conv patch compression
        input_ids = input_ids[:max_length - n_images * per_image_length]
        labels = labels[:max_length - n_images * per_image_length]
    
    # Prepare the result dictionary with tokenized data
    result = {'input_ids': input_ids, 'labels': labels}
    
    return result


class Packer:
    """Pack multiple pieces of data into one."""

    def __init__(self,
                 chunk_size=2048,
                 use_varlen_attn=False,
                 drop_last=False):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}
        self.use_varlen_attn = use_varlen_attn
        self.drop_last = drop_last
        if use_varlen_attn:
            self.residual_cumulative_len = [0]

    def get_cumulative_len(self, chunk_num):
        ptr_l = 0
        cumulative_len = []
        for chunk_idx in range(chunk_num):
            length_train = (chunk_idx + 1) * self.chunk_size
            ptr_r = np.searchsorted(
                self.residual_cumulative_len, length_train, side='left')
            if self.residual_cumulative_len[ptr_r] == length_train:
                cumulative_len_cur = \
                    self.residual_cumulative_len[ptr_l:ptr_r + 1]
                ptr_l = ptr_r + 1
            else:
                cumulative_len_cur = self.residual_cumulative_len[
                    ptr_l:ptr_r] + [length_train]
                ptr_l = ptr_r
            cumulative_len_cur = [
                num - chunk_idx * self.chunk_size for num in cumulative_len_cur
            ]
            if cumulative_len_cur[0] != 0:
                cumulative_len_cur = [0] + cumulative_len_cur

            cumulative_len.append(cumulative_len_cur)

        self.residual_cumulative_len = [
            num - length_train for num in self.residual_cumulative_len[ptr_l:]
        ]
        if len(self.residual_cumulative_len) == 0:
            self.residual_cumulative_len = [0]
        elif self.residual_cumulative_len[0] != 0:
            self.residual_cumulative_len = [0] + self.residual_cumulative_len

        return cumulative_len

    def get_position_ids(self, cumulative_len):
        position_ids = []
        for cumulative_len_cur in cumulative_len:
            index_cur = []
            for i in range(len(cumulative_len_cur) - 1):
                index_cur.extend(
                    list(
                        range(cumulative_len_cur[i + 1] -  # noqa: W504
                              cumulative_len_cur[i])))
            position_ids.append(index_cur)
        return position_ids

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        if self.use_varlen_attn:
            for input_id in batch['input_ids']:
                self.residual_cumulative_len.append(
                    self.residual_cumulative_len[-1] + len(input_id))

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size] for i in range(
                        0,
                        chunk_num *  # noqa: W504
                        self.chunk_size,
                        self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }

            if self.use_varlen_attn:
                cumulative_len = self.get_cumulative_len(chunk_num)
                result['cumulative_len'] = cumulative_len
                result['position_ids'] = self.get_position_ids(cumulative_len)
        else:
            if self.drop_last:
                result = {k: [] for k, v in concatenated_samples.items()}
            else:
                result = {k: [v] for k, v in concatenated_samples.items()}

            self.residual = {k: [] for k in concatenated_samples.keys()}

            if self.use_varlen_attn:
                result['cumulative_len'] = [] if self.drop_last else [
                    self.residual_cumulative_len
                ]
                result['position_ids'] = [] if self.drop_last \
                    else self.get_position_ids([self.residual_cumulative_len])
                self.residual_cumulative_len = [0]

        return result

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class PadToGrid:
    """
    Pads sparse features into a dense feature grid.
    """
    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, sample):
        """
        Args:
            sample (tuple): A tuple containing (features, coords, patch_size).

        Returns:
            torch.Tensor: Dense grid of features with shape (C, H, W).
        """
        features, coords, patch_size = sample
        features = torch.as_tensor(features, dtype=torch.float32)
        coords = torch.as_tensor(coords, dtype=torch.long)

        # Convert absolute coordinates to grid coordinates
        grid_coords = coords // patch_size
        
        # Find the bounding box of the occupied grid cells
        min_coords = torch.min(grid_coords, dim=0).values
        max_coords = torch.max(grid_coords, dim=0).values
        
        # Shift coordinates to be relative to the top-left corner of the bounding box
        shifted_coords = grid_coords - min_coords
        
        # Calculate the dimensions of the final dense grid
        grid_dims = max_coords - min_coords + 1
        grid_h, grid_w = grid_dims[1].item(), grid_dims[0].item()

        feature_dim = features.shape[1]
        
        # Create a dense feature grid filled with the padding value
        # Use (H, W, C) for easier indexing, then permute
        feature_grid = torch.full(
            (grid_h, grid_w, feature_dim), 
            fill_value=self.pad_value, 
            dtype=features.dtype
        )
        
        # Place the features into the feature grid at their respective locations
        feature_grid[shifted_coords[:, 1], shifted_coords[:, 0]] = features

        # Permute feature grid to (C, H, W)
        return feature_grid.permute(2, 0, 1)


class CenterFixedSizeCrop:
    """
    Performs a center crop with a fixed output size.
    If the image is smaller than the crop size, it is padded first.
    If the center crop contains no features, the nearest non-empty region is
    used instead. The output is always ``(C, crop_h, crop_w)``.
    """
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Feature grid of shape (C, H, W).
        Returns:
            torch.Tensor: Fixed-size cropped feature grid.
        """
        grid = sample
        _, h, w = grid.shape
        th, tw = self.crop_size

        # Pad if needed
        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            grid = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0)

        _, h, w = grid.shape
        i = (h - th) // 2
        j = (w - tw) // 2
        crop = grid[:, i:i + th, j:j + tw]

        # A sparse WSI may have an empty geometric center. Anchor the crop at
        # the non-empty patch nearest the grid center, while keeping its size
        # bounded by (th, tw).
        occupied = grid.abs().sum(dim=0).ne(0)
        if not occupied[i:i + th, j:j + tw].any() and occupied.any():
            coords = occupied.nonzero(as_tuple=False)
            center_y = (h - 1) / 2
            center_x = (w - 1) / 2
            distance = (coords[:, 0].float() - center_y).square()
            distance += (coords[:, 1].float() - center_x).square()
            anchor_y, anchor_x = coords[distance.argmin()].tolist()
            i = min(max(anchor_y - th // 2, 0), h - th)
            j = min(max(anchor_x - tw // 2, 0), w - tw)
            crop = grid[:, i:i + th, j:j + tw]

        return crop


class RandomVariableCrop:
    """
    Performs a random crop with a variable output size.
    """
    def __init__(self, scale=(0.9, 1.0), ratio=(0.2, 5.0)):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Feature grid of shape (C, H, W).
        Returns:
            torch.Tensor: Cropped feature grid.
        """
        grid = sample
        
        # Use the library's robust function to get crop parameters based on the grid's size
        top, left, h, w = transforms.RandomResizedCrop.get_params(grid, self.scale, self.ratio)
        
        return grid[:, top:top + h, left:left + w]


class RandomVariableCropWithLimit:
    """
    Performs a random crop with a variable output size, with an upper limit on the total number of patches.
    1. Applies RandomVariableCrop.
    2. If the resulting number of patches exceeds max_patch_num, crops again to meet the limit.
    """
    def __init__(self, scale=(0.7, 1.0), ratio=(0.2, 5.0), max_patch_num=20000):
        self.scale = scale
        self.ratio = ratio
        self.max_patch_num = max_patch_num

    def __call__(self, sample):
        grid = sample
        
        # 1. Variable Crop
        top, left, h, w = transforms.RandomResizedCrop.get_params(grid, self.scale, self.ratio)
        grid = grid[:, top:top + h, left:left + w]
        
        # 2. Check limit and crop again if needed
        _, h, w = grid.shape
        if h * w > self.max_patch_num:
            # Calculate new dimensions to fit max_patch_num while maintaining aspect ratio
            current_ratio = w / h
            new_h = int(np.sqrt(self.max_patch_num / current_ratio))
            new_w = int(new_h * current_ratio)
            
            # Ensure we don't exceed current dimensions
            new_h = min(new_h, h)
            new_w = min(new_w, w)
            
            # Random crop to new size
            i = torch.randint(0, h - new_h + 1, size=(1, )).item()
            j = torch.randint(0, w - new_w + 1, size=(1, )).item()
            
            grid = grid[:, i:i+new_h, j:j+new_w]

        return grid

def load_wsi_feature(wsi_file, max_patch_num, transform=None):
    with h5py.File(wsi_file, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:]
        patch_size = f['coords'].attrs.get('patch_size_level0', 512)

    # do random sampling
    if max_patch_num is not None and max_patch_num > 0:   
        total_patches = features.shape[0]
        if total_patches >= max_patch_num:
            indices = np.linspace(0, total_patches - 1, max_patch_num, dtype=int)
            features = features[indices]
            coords = coords[indices]
    
    # do padding and random crop
    sample = (features, coords, patch_size)
    if transform:
        return transform(sample)
    else:
        features = torch.from_numpy(features)
        return features


def load_wsi_global_features(wsi_feature_paths, feature_key='feature'):
    """
    Load global WSI-level feature vectors from a list of H5 files.
    
    This function loads pre-extracted slide-level embeddings from multiple
    WSI feature extractors (e.g., TITAN, CONCH, UNI).
    
    Args:
        wsi_feature_paths: List of paths to H5 files containing WSI features.
        feature_key: Primary key to look for in H5 files. Will also try common
                     alternative keys if the primary key is not found.
    
    Returns:
        List of 1D tensors, one for each WSI feature file.
        
    Example:
        >>> paths = [
        ...     '/path/to/wsi_feat_titan.h5',
        ...     '/path/to/wsi_feat_conch.h5',
        ...     '/path/to/wsi_feat_uni.h5'
        ... ]
        >>> features = load_wsi_global_features(paths)
        >>> [f.shape for f in features]
        [torch.Size([768]), torch.Size([1024]), torch.Size([768])]
    """
    if not wsi_feature_paths:
        return []
    
    # Common key names used by different WSI feature extractors
    possible_keys = [
        feature_key,
        'feature',
        'features',
        'embedding',
        'embeddings',
        'feat',
        'slide_feature',
        'wsi_feature',
        'global_feature',
    ]
    
    features = []
    for path in wsi_feature_paths:
        feat = _load_single_wsi_feature(path, possible_keys)
        features.append(feat)
    
    return features


def _load_single_wsi_feature(h5_path, possible_keys):
    """Load a single WSI feature from an H5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Try each possible key
        for key in possible_keys:
            if key in f:
                data = f[key][:]
                
                # Handle different array shapes
                if data.ndim == 0:
                    raise ValueError(f"Feature in {h5_path} is scalar, expected 1D array")
                elif data.ndim == 1:
                    pass  # Already 1D
                elif data.ndim == 2:
                    if data.shape[0] == 1:
                        data = data.squeeze(0)
                    else:
                        # Multiple features - take mean for aggregation
                        data = data.mean(axis=0)
                else:
                    data = data.reshape(-1)
                
                return torch.from_numpy(data.astype(np.float32))
        
        # No valid key found
        available_keys = list(f.keys())
        raise KeyError(
            f"No valid feature key found in {h5_path}. "
            f"Available keys: {available_keys}. "
            f"Tried: {possible_keys[:5]}..."
        )
