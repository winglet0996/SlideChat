# usage
- The training of this project is launched with slurm by `sbatch run_xtuner_train_conv_multi_gpu.sh`.
- Use `/mnt/petrelfs/zhouxiao/anaconda3/envs/slidechat/bin/python` for project Python commands and tests.

# xtuner architecture
- xtuner is a Python config/registry style based LLM/VLM finetuning framework.
- The training or testing setting is managed with configs under `SlideChat/xtuner/configs/slidechat`, mainly `stage_2_qwen3_8b_conv_multitask_qwen35_vl_patch_wsi.py` and `stage_2_qwen3_8b_conv_multitask_qwen3_vl.py` series.
- The models are defined under `SlideChat/xtuner/model`, mainly `llava_qwen3_5.py` and `llava_conv_qwen3_vl.py`.
- The data pipeline is defined under `SlideChat/xtuner/dataset`, mainly `llava_dataset.py`; and text preprocessing fns at `SlideChat/xtuner/dataset/map_fns/dataset_map_fns/llava_map_fn.py` and `SlideChat/xtuner/dataset/collate_fns/masked_collated_fn.py`.
- Training and testing modular engines are under `SlideChat/xtuner/tools`, `train.py` and `test.py`.

# Qwen3.5 prompt-conditioned patch model notes
- `xtuner/model/llava_qwen3_5.py` implements the Qwen3.5 multimodal multitask model used by the `stage_2_qwen3_8b_conv_multitask_qwen35_vl_patch_wsi.py` config.
- Patch features are converted into visual tokens by `PromptConditionedPatchResampler` in `xtuner/model/custom_model.py`. The resampler conditions patch selection on text prompt context.
- `prompt_context_mode='llm_hidden'` extracts prompt context by running the active LLM language model on prompt-only tokens. When LoRA is enabled this path should use the LoRA-wrapped LLM, not a base-model-only context. `prompt_context_detach=True` means the prompt-context forward is run under no-grad/eval for stability, but it still reads the current LoRA weights.
- Qwen3.5 uses linear-attention layers whose mask code assumes left padding. The composed LLM inputs in `llava_qwen3_5.py` should therefore be left padded for training and prediction. Do not switch the multimodal composed batch back to right padding without revalidating Qwen3.5 linear-attention masking.
- Newly initialized trainable modules around the LLM, especially `PromptConditionedPatchResampler`, `WSIProjector`, `RegressionHead`, and `SurvivalHead`, should stay in FP32 even when the pretrained LLM runs in BF16. Casting these fresh trainable modules to BF16 can make AdamW updates numerically fragile.
- Do not train the full input embedding or `lm_head.weight` matrices just to adapt `<REG>` / `<SRV>` tokens. They are huge BF16 vocab matrices and can become non-finite under AdamW. Initialize the special-token rows, then keep the full vocab embedding/head weights frozen; LoRA and task heads should carry the training signal.
- LM loss is computed only on active supervised labels and with logits upcast to FP32 before cross entropy. Ignored labels should not participate in CE.
- Multitask batches can contain placeholder NaN targets for tasks absent from a sample. Regression loss must filter to finite regression targets before `SmoothL1Loss`; otherwise mixed batches can produce `reg_loss=nan`.
