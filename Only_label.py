import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
import json
import re
import copy
import gc



def process_func(example):
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "Please judge if the image is harmful"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本

    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}

tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct", use_fast=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct")

stage_1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/output/stage-1/checkpoint-472", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen2.5-VL-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)

base_model.visual = copy.deepcopy(stage_1_model.visual)



base_model.enable_input_require_grads() 

stage_1_model = None

del stage_1_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


train_json_path = "/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/label_train.json"

#都当做训练集
train_ds = Dataset.from_json(train_json_path)
train_dataset = train_ds.map(process_func)


# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj","down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
    modules_to_save = ['visual','model.lm_head']
)

# 获取LoRA模型
peft_model = get_peft_model(base_model, config)

for n,p in peft_model.visual.named_parameters():
    p.requires_grad = True

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/stage-2-label",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=300,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    # use_cache = False
)
        

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 开启模型训练
trainer.train()

