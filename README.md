# <img src="./img/icon.jpg" width="70" height="63"> SlideChat

[🍎 **Homepage**](https://uni-medical.github.io/SlideChat.github.io/) | [🤗 **Dataset**](https://huggingface.co/datasets/General-Medical-AI/SlideBench) | [📖**Paper**](https://arxiv.org/pdf/2410.11761v1) 

This repository is the official code base of the paper **SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding (CVPR2025)**.

# Getting Start


This project is built upon [**Xtuner**](https://github.com/InternLM/xtuner). To get started:

```bash
git clone https://github.com/uni-medical/SlideChat.git
cd SlideChat
pip install -e .
```

## Train SlideChat:
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
