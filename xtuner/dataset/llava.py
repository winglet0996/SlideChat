# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
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


class LLaVADataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_path_list,
                 per_image_length,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 sample_num=10240):
        super().__init__()

        self.sample_num = sample_num
        self.per_image_length = per_image_length
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
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError

            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            self.text_data = process_hf_dataset(
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
                per_image_length=self.per_image_length)

        self.image_folder = image_folder
        self.image_path_list = image_path_list
        self.pad_image_to_square = pad_image_to_square

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
                        # coords = f['coords'][:]

                    total_rows = image.shape[0]
                    if total_rows >= self.sample_num:
                        indices = np.linspace(0, total_rows - 1, self.sample_num, dtype=int)
                        sampled_df = image.iloc[indices]
                        image = sampled_df.iloc[:self.sample_num]
                    
                    image = image.to_numpy()
                    image = torch.from_numpy(image)
                images.append(image)
            data_dict['pixel_values'] = images
        return data_dict
