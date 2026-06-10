# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
import logging
import os
import shutil
import time

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from .huggingface import process_hf_dataset
from .llava_dataset import (DATASET_CACHE_VERSION, _cache_lock,
                            _json_safe_signature, _path_signature)
from .utils import expand2square

import pandas as pd
import h5py
import numpy as np

def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def _build_cache_signature(data_path,
                           tokenizer,
                           max_dataset_length,
                           dataset_map_fn,
                           template_map_fn,
                           max_length,
                           per_image_length,
                           max_patch_num,
                           input_ids_with_output):
    return {
        'cache_version': DATASET_CACHE_VERSION,
        'dataset_type': 'LLaVADataset_longnet',
        'data_path': _path_signature(data_path),
        'tokenizer': _json_safe_signature(tokenizer),
        'max_dataset_length': max_dataset_length,
        'dataset_map_fn': _json_safe_signature(dataset_map_fn),
        'template_map_fn': _json_safe_signature(template_map_fn),
        'max_length': max_length,
        'per_image_length': per_image_length,
        'max_patch_num': max_patch_num,
        'input_ids_with_output': input_ids_with_output,
        'dataset_processor': _json_safe_signature(process_hf_dataset),
    }


class LLaVADataset_longnet(Dataset):

    def __init__(self,
                 image_folder,
                 image_path_list,
                 per_image_length,
                 data_path=None,
                 cache_dir=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=None,
                 pad_image_to_square=False,
                 max_patch_num=None,
                 mode=None,
                 input_ids_with_output=True):
        super().__init__()

        self.max_patch_num = max_patch_num
        self.per_image_length = per_image_length
        self.mode = mode
        assert offline_processed_text_folder or (data_path and tokenizer)
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            self.text_data = self._build_or_load_text_data(
                data_path=data_path,
                cache_dir=cache_dir,
                tokenizer=tokenizer,
                max_dataset_length=max_dataset_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                max_length=max_length,
                input_ids_with_output=input_ids_with_output)

        self.image_folder = image_folder
        self.image_path_list = image_path_list
        self.pad_image_to_square = pad_image_to_square

    def _build_or_load_text_data(self,
                                 data_path,
                                 cache_dir,
                                 tokenizer,
                                 max_dataset_length,
                                 dataset_map_fn,
                                 template_map_fn,
                                 max_length,
                                 input_ids_with_output):
        cache_signature = None
        cache_path = None
        cache_dataset_path = None
        cache_meta_path = None
        lock_path = None

        if cache_dir:
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            cache_signature = _build_cache_signature(
                data_path=data_path,
                tokenizer=tokenizer,
                max_dataset_length=max_dataset_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                max_length=max_length,
                per_image_length=self.per_image_length,
                max_patch_num=self.max_patch_num,
                input_ids_with_output=input_ids_with_output)
            cache_key = hashlib.sha256(
                json.dumps(
                    cache_signature,
                    sort_keys=True,
                    ensure_ascii=False).encode('utf-8')).hexdigest()
            cache_path = os.path.join(cache_dir, cache_key)
            cache_dataset_path = os.path.join(cache_path, 'dataset')
            cache_meta_path = os.path.join(cache_path, 'meta.json')
            lock_path = os.path.join(cache_dir, f'{cache_key}.lock')

            cached_dataset = self._try_load_cached_dataset(
                cache_dataset_path=cache_dataset_path,
                cache_meta_path=cache_meta_path,
                cache_signature=cache_signature)
            if cached_dataset is not None:
                return cached_dataset

        processed_dataset = self._process_text_data(
            data_path=data_path,
            tokenizer=tokenizer,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            input_ids_with_output=input_ids_with_output)

        if cache_path is None:
            return processed_dataset

        with _cache_lock(lock_path):
            cached_dataset = self._try_load_cached_dataset(
                cache_dataset_path=cache_dataset_path,
                cache_meta_path=cache_meta_path,
                cache_signature=cache_signature)
            if cached_dataset is not None:
                return cached_dataset

            tmp_cache_path = os.path.join(
                os.path.dirname(cache_path),
                f'.tmp_{os.path.basename(cache_path)}_{os.getpid()}_{time.time_ns()}')
            tmp_dataset_path = os.path.join(tmp_cache_path, 'dataset')
            tmp_meta_path = os.path.join(tmp_cache_path, 'meta.json')

            try:
                os.makedirs(tmp_cache_path, exist_ok=False)
                processed_dataset.save_to_disk(tmp_dataset_path)
                with open(tmp_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_signature, f, indent=2, ensure_ascii=False)
                os.rename(tmp_cache_path, cache_path)
                print_log(
                    f'Saved processed dataset cache to {cache_path}',
                    logger='current')
            except FileExistsError:
                print_log(
                    f'Dataset cache was created concurrently, reusing {cache_path}',
                    logger='current',
                    level=logging.WARNING)
            finally:
                shutil.rmtree(tmp_cache_path, ignore_errors=True)

        cached_dataset = self._try_load_cached_dataset(
            cache_dataset_path=cache_dataset_path,
            cache_meta_path=cache_meta_path,
            cache_signature=cache_signature)
        return cached_dataset or processed_dataset

    def _try_load_cached_dataset(self,
                                 cache_dataset_path,
                                 cache_meta_path,
                                 cache_signature):
        if not cache_dataset_path or not cache_meta_path:
            return None

        if not (os.path.isdir(cache_dataset_path)
                and os.path.isfile(cache_meta_path)):
            return None

        try:
            with open(cache_meta_path, 'r', encoding='utf-8') as f:
                cached_signature = json.load(f)
            if cached_signature != cache_signature:
                return None

            print_log(
                f'Loading processed dataset cache from {os.path.dirname(cache_dataset_path)}',
                logger='current')
            return load_from_disk(cache_dataset_path)
        except Exception as e:
            print_log(
                f'Failed to load dataset cache from {os.path.dirname(cache_dataset_path)}: {e}. '
                'The cache will be rebuilt.',
                logger='current',
                level=logging.WARNING)
            shutil.rmtree(os.path.dirname(cache_dataset_path), ignore_errors=True)
            return None

    def _process_text_data(self,
                           data_path,
                           tokenizer,
                           max_dataset_length,
                           dataset_map_fn,
                           template_map_fn,
                           max_length,
                           input_ids_with_output):
        if data_path.endswith('.json'):
            with open(data_path, encoding='utf-8') as f:
                json_data = json.load(f)
        elif data_path.endswith('.jsonl'):
            json_data = load_jsonl(data_path)
        else:
            raise NotImplementedError

        for idx in range(len(json_data)):
            if isinstance(json_data[idx]['id'], int):
                json_data[idx]['id'] = str(json_data[idx]['id'])

        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        return process_hf_dataset(
            dataset=json_data,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            split='train',
            max_dataset_length=max_dataset_length,
            remove_unused_columns=False,
            pack_to_max_length=False,
            with_image_token=True,
            per_image_length=self.per_image_length,
            max_patch_num=self.max_patch_num,
            input_ids_with_output=input_ids_with_output)

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            image = data_dict.get('image', None)
            if image is None:
                cur_len = -cur_len
            else:
                if isinstance(image, str):
                    n_images = 1
                else:
                    n_images = len(image)
                cur_len = cur_len - n_images + self.per_image_length * n_images
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_list = data_dict['image']
            if isinstance(image_list, str):
                image_list = [image_list]
            images = []
            for image_file in image_list:
                if image_file.endswith('.h5'):

                    with h5py.File(image_file, 'r') as f:
                        image = f['features'][:]

                    if image.ndim == 2 and image.shape[1] > 512:
                        image = image[:, :512]

                    total_rows = image.shape[0]
                    if total_rows >= self.max_patch_num:
                        indices = np.linspace(0, total_rows - 1, self.max_patch_num, dtype=int)
                        image = image[indices]
                    image = torch.from_numpy(image)
                images.append(image)
            data_dict['pixel_values'] = images
        return data_dict
