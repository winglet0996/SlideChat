# SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding (CVPR2025)

[🍎 **Homepage**](https://uni-medical.github.io/SlideChat.github.io/) | [🤗 **Model and Dataset**](https://huggingface.co/datasets/General-Medical-AI) | [📖**Paper**](https://arxiv.org/pdf/2410.11761v1) 

Despite the progress made by multimodal large language models (MLLMs) in computational pathology, they remain limited by a predominant focus on patchlevel analysis, missing essential contextual information at the whole-slide level. The lack of large-scale instruction datasets and the gigapixel scale of whole slide images (WSIs) pose significant developmental challenges. In this paper, we present SlideChat, the first vision-language assistant capable of understanding gigapixel whole-slide images, exhibiting excellent multimodal conversational capability and response complex instruction across diverse pathology scenarios. To support its development, we created SlideInstruction, the largest instructionfollowing dataset for WSIs consisting of 4.2K WSI captions and 176K VQA pairs with multiple categories. Furthermore, we propose SlideBench, a multimodal benchmark that incorporates captioning and VQA tasks to assess SlideChat’s capabilities in varied clinical settings such as microscopy, diagnosis. Compared to both general and specialized MLLMs, SlideChat exhibits exceptional capabilities, achieving state-of-the-art performance on 18 of 22 tasks. 

<p align="center">
    <img src="img/Fig1_slidechat_illustration.png" width="80%"> <br>
</p>

## Release
We release **SlideChat**, **SlideInstruction**, and **SlideBench** as open-source resources, hoping to facilitate research and development in computational pathology.
- **SlideChat**: The first large vision-language assistant for whole-slide pathology image analysis, capable of generating comprehensive descriptions and contextually relevant responses.
- **SlideInstruction**: The largest comprehensive WSI instruction-following dataset, derived from pathology reports..
- **SlideBench**: A WSI multimodal benchmark including SlideBench-Caption, SlideBench-VQA (TCGA), and SlideBench-VQA (BCNB).  Before its final open-sourcing, SlideBench underwent a second round of expert review and filtering in collaboration with pathologists to ensure data quality.

# Usage

This project is built upon [**Xtuner**](https://github.com/InternLM/xtuner). To get started:

```bash
git clone https://github.com/uni-medical/SlideChat.git
cd SlideChat
pip install -e .
```

## Pre-requisites:
We share our dataset but WSIs still need to be preprocessed due to their large resolution. For a quick start, we provide several WSI features after processing in the repository. You can now download our code and directly run the code.

Downloading TCGA Slides
To download diagnostic WSIs (formatted as .svs files), please refer to the NIH Genomic Data Commons Data Portal. WSIs for each cancer type can be downloaded using the GDC Data Transfer Tool.

Processing Whole Slide Images
To process WSIs, first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained backbone is used to encode raw image patches into feature vectors, which we then save as .pt files for each WSI. We achieve the pre-processing of WSIs by using CLAM


## Train:
```bash
xtuner train \
 <your config file path>  \
  --deepspeed <deepspeed config file path> \
  --work-dir <workdir path>

# For example
xtuner train \
 configs/slidechat/stage_1.py \
  --deepspeed configs/deepspeed/deepspeed_zero2.json \
  --work-dir work_dirs/stage1
```


## Inference

```bash
xtuner test <your config file path> \
--checkpoint <your checkpoint path> \
--test_image_csv  my_test_Conversation_2.csv \
--test_output_csv output_my_test_Conversation_3.csv \
--local_rank 0
```

# Config file
Config files are in `configs/`.
## Explanation of the config file.
For a detailed explanation of the configuration file, please refer [**here**](https://xtuner.readthedocs.io/zh-cn/latest/training/modify_settings.html).

- `llm_name_or_path`: The parameter `llm_name_or_path` corresponds to the Hugging Face LLM path, such as `internlm/internlm2-chat-7b` or `Qwen/Qwen2.5-0.5B-Instruct` and so on.
- `data_path` and `image_path_list`: Training data path(refer our config file).
- `evaluation_images`: Evaluation data path.

# LLAVAModel
## Hyperparameters
- `freeze_llm`: Freeze the parameters of the LLM.
- `freeze_visual_encoder`: Freeze the parameters of the visual encoder.
- `pretrained_pth`: If it is the stage 2 training , it refers to the checkpoint file from stage 1 training; otherwise, it is set to `None`.
- `train_stage`: `train_stage` indicates the training phase, either Stage `'1'` or Stage `'2'`.

## Contact

- Yuanfeng Ji: yfj@stanford.edu
- Junjun He: hejunjun@pjlab.org.cn

## Citation

**BibTeX:**

```bibtex
@article{chen2024slidechat,
  title={SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding},
  author={Chen, Ying and Wang, Guoan and Ji, Yuanfeng and Li, Yanjun and Ye, Jin and Li, Tianbin and Zhang, Bin and Pei, Nana and Yu, Rongshan and Qiao, Yu and others},
  journal={arXiv preprint arXiv:2410.11761},
  year={2024}
}
```
