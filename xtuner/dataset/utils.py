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
            'QWenTokenizer', 'QWen2Tokenizer', 'Qwen2TokenizerFast'
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
    return {'input_ids': input_ids, 'labels': labels}


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
    Pads sparse features into a dense grid and generates a corresponding mask.
    The mask indicates valid (1) vs. padded (0) areas, which is required
    by downstream modules like Partial Convolution (PConv).
    """
    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, sample):
        """
        Args:
            sample (tuple): A tuple containing (features, coords, patch_size).

        Returns:
            tuple: A tuple containing (feature_grid, mask).
                   - feature_grid (torch.Tensor): The dense grid of features (C, H, W).
                   - mask (torch.Tensor): The binary mask (1, H, W).
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
        
        # Create a corresponding mask grid initialized to zeros (padded)
        mask_grid = torch.zeros((grid_h, grid_w), dtype=torch.float32)

        # Place the features into the feature grid at their respective locations
        feature_grid[shifted_coords[:, 1], shifted_coords[:, 0]] = features
        
        # Set the mask to 1.0 at the locations of valid features
        mask_grid[shifted_coords[:, 1], shifted_coords[:, 0]] = 1.0
        
        # Permute feature grid to (C, H, W) and add channel dim to mask to get (1, H, W)
        return feature_grid.permute(2, 0, 1), mask_grid.unsqueeze(0)

class RandomVariableCrop:
    """
    Performs a random crop with a variable output size.
    Crucially, it applies the *same* crop to both the feature grid and its mask
    to maintain their correspondence.
    """
    def __init__(self, scale=(0.75, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        """
        Args:
            sample (tuple): A tuple containing (grid, mask).
                            - grid (torch.Tensor): A feature grid of shape (C, H, W).
                            - mask (torch.Tensor): A mask of shape (1, H, W).
        Returns:
            tuple: A tuple containing the cropped (grid, mask).
        """
        grid, mask = sample
        
        # Use the library's robust function to get crop parameters based on the grid's size
        top, left, h, w = transforms.RandomResizedCrop.get_params(grid, self.scale, self.ratio)
        
        # Apply the exact same crop to both the grid and the mask
        cropped_grid = grid[:, top:top + h, left:left + w]
        cropped_mask = mask[:, top:top + h, left:left + w]
        
        return cropped_grid, cropped_mask

def load_wsi_feature(wsi_file, max_patch_num, transform=None):
    with h5py.File(wsi_file, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:]
        patch_size = f['coords'].attrs.get('patch_size', 256)

    # do random sampling
    if max_patch_num is not None and max_patch_num > 0:   
        total_patches = features.shape[0]
        if total_patches >= max_patch_num:
            indices = np.linspace(0, total_patches - 1, max_patch_num, dtype=int)
            features = features[indices]
    
    # do padding and random crop
    sample = (features, coords, patch_size)
    if transform:
        return transform(sample)
    else:
        features = torch.from_numpy(features)
        return features

