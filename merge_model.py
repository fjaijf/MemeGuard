from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType
import re
import json
import torch
import argparse
import os
import copy

# config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         inference_mode=True,
#         r=64,  # Lora 秩
#         lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#         lora_dropout=0.05,  # Dropout 比例
#         bias="none",
#         modules_to_save = ['visual','model.lm_head']
#     )


# tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct", use_fast=True, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct")

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-2", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)

# stage_3_model = PeftModel.from_pretrained(model, model_id="/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/output/stage-3/continue-1/checkpoint-945", config=config)
# print(stage_3_model)

# stage_3_model = stage_3_model.merge_and_unload()

# stage_3_model.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-3.1")
# processor.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-3.1")
# tokenizer.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-3.1")



# config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         inference_mode=True,
#         r=64,  # Lora 秩
#         lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#         lora_dropout=0.05,  # Dropout 比例
#         bias="none",
#         modules_to_save = ['visual','model.lm_head']
#     )


# tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct", use_fast=True, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct")

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)

# stage_2_model = PeftModel.from_pretrained(model, model_id="/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/output/stage-2-label/checkpoint-944", config=config)
# print(stage_2_model)

# stage_2_model = stage_2_model.merge_and_unload()

# stage_2_model.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/label")
# processor.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/label")
# tokenizer.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/label")


config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
        modules_to_save = ['visual','model.lm_head']
    )


tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-adj2", use_fast=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-adj2")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-adj2", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)

stage_2_model = PeftModel.from_pretrained(model, model_id="/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/output/ablation/stage-3/continue/checkpoint-945", config=config)
print(stage_2_model)

stage_2_model = stage_2_model.merge_and_unload()

stage_2_model.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-adj3")
processor.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-adj3")
tokenizer.save_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-adj3")
