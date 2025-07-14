from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType
import re
import json
import torch
import argparse
import os
from tqdm import tqdm


if __name__ == "__main__":
    save = []
    label_list = []
    result_list = []
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

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-3", torch_dtype="auto", device_map="auto"
    )
    # model = PeftModel.from_pretrained(model, model_id="/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/output/stage-3-3/Qwen2.5-VL-7B-CoT-FHM/checkpoint-607", config=config)

    processor = AutoProcessor.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-3")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/save_model/stage-3", use_fast=False, trust_remote_code=True)
    accurate = 0
    with open("/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/LLaMA-Factory/test.json", "r") as f:
        dataset = json.load(f)
    for i in tqdm(range(len(dataset))):
        origin_image_path = dataset[i]['images'][0]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": origin_image_path
                    },
                    {"type": "text", 
                    "text": "Please judge if this image is harmful "}
                    #  "text": "Please judge if the image is harmful and follow the steps bleow :In SUMMARY, briefly explain what steps you’ll take to solve the problem. In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. In JUDGEMENT, give the final judgement based on the information. The judgement must be stictly in the format:'The image's label is harmful.' or 'The image's label is harmless.'"}
                    #  "text": "Now you need to explain and classify some harmful content, which can be divided into four aspects: 1. Discrimination: including discriminatory content based on race, gender, religion, nationality, LGBTQ+community, etc. For example, stereotypes, denigration of specific groups, racism, gender discrimination, etc. 2. Offensive: Content that insults, belittles, or maliciously attacks individuals or groups. For example: personal attacks, insults, insulting language, malicious satire, etc. 3. Violence: Content that directly or indirectly encourages or depicts violent behavior. For example, involving killing, war, abuse, terrorism, bloody scenes, etc. 4. Vulgar: Content that contains sexual innuendos, obscenities, nudity, and vulgar tastes. For example: pornographic content, extremely vulgar language, excessive nudity, indecent behavior, etc. 5. Dissatisfaction: Includes content that expresses discontent, resentment, or negative emotions toward individuals, groups, or society through satire, sarcasm, or exaggeration.Examples: extreme sarcasm, cynical mockery, deliberate denigration of social systems, excessive portrayal of negative emotions, spreading pessimistic or nihilistic views. Please answer based on these five aspects.Here's how the format should look: <SUMMARY> [Summarize and analyze what aspects need to be analyzed when judging whether it is harmful or not.] </SUMMARY> <CAPTION> [extract the meaning behind the text description, including background information, character information, and homophonic information.] </CAPTION> <REASONING> [Provide a chain-of-thought, logical explanation based on the above information. This should outline step-by-step reasoning.] </REASONING> <JUDGEMENT> [Use prior knowledge to give your own judgment with answer only 'This picture is harmful/harmless'] </JUDGEMENT> (Do not forget </JUDGEMENT>!) Please apply this format meticulously to analyze the image and answer the related question. According to the image,reason according to four parts based on CoT technology: 1. Summarize and analyze what aspects need to be analyzed when making judgments. 2. Extract the meaning behind the text description, including background information(culture,country,ethnic and so on), character information, homophonic information, the relationship of characters, the relationship of characters and background.  3. Make detailed reasoning based on the above information. 4.Give the label of this image with your priori knowledge with answer only 'This picture is harmful/harmless'"},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1000,use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        #label
        label = dataset[i]["conversations"][1]['value']
        if label == 'nonharmful' or label == 'harmless':
            label = 'harmless'
        else:
            label = 'harmful'
        label_list.append(label)

        if output_text[0] == 'nonharmful' or output_text[0] == 'harmless':
            result = 'harmless'
        else:
            result = 'harmful'
        
        result_list.append(result)

        if label == result:
            accurate += 1
            save.append({'image':origin_image_path,'pred_result':'correct'})
        else:
            save.append({'image':origin_image_path,'pred_result':'wrong'})
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(label_list)):
        if label_list[i] == 'harmful' and result_list[i] == 'harmful':
            TP += 1
        if label_list[i] == 'harmless' and result_list[i] == 'harmless':
            TN += 1
        if label_list[i] == 'harmless' and result_list[i] == 'harmful':
            FP += 1
        if label_list[i] == 'harmful' and result_list[i] == 'harmless':
            FN += 1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*((P*R)/(P+R))
    print("P:",P)
    print("R:",R)
    print("F1:",F1)
    print('accuracy: ',accurate/len(dataset))

        
            
with open('/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/result.json','w',encoding='utf8') as f:
    json.dump(save,f,indent=4,ensure_ascii=False)        
