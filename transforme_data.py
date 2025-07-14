import json

save_path = '/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/LLaMA-Factory/CoT_test.json'
with open('/mnt/pfs-guan-ssai/nlu/guhexiang/Qwen-CoT/CoT_test.json','r',encoding='utf8') as f:
    data = json.load(f)

data_copy = []
for i in range(len(data)):
    tmp_dict = {}
    conversations = []
    images = []
    human = {}
    gpt = {}
    human["from"] = "human"
    human["value"] = "<image>Please judge if the image is harmful and follow the steps bleow :In SUMMARY, briefly explain what steps you’ll take to solve the problem. In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. In JUDGEMENT, give the final answer"
    conversations.append(human)
    gpt["from"] = "gpt"
    gpt["value"] = data[i]["conversations"][1]["value"]
    conversations.append(gpt)
    tmp_dict["conversations"] = conversations
    images.append(data[i]["conversations"][0]["value"].split("<|vision_start|>")[1].split("<|vision_end|>")[0] )
    tmp_dict["images"] = images
    data_copy.append(tmp_dict)

with open(save_path,'w',encoding='utf8') as f:
    json.dump(data_copy,f,indent=4,ensure_ascii=False)
