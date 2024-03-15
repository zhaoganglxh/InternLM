from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
#model_dir = snapshot_download("J:\HuggingFace\models\Shanghai_AI_Laboratory\internlm-7b", revision='v1.0.2')
model_dir = "J:\HuggingFace\models\Shanghai_AI_Laboratory\internlm-chat-7b-v1_1"
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True)
# `torch_dtype=torch.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，有可能导致显存不足
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="cuda",  trust_remote_code=True, torch_dtype=torch.float16).cuda()
model = model.eval()

while True:
    inputText = input("请输入内容：")
    # inputs = tokenizer([inputText], return_tensors="pt")
    # for k,v in inputs.items():
    #     inputs[k] = v.cuda()
    # gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.1}
    # output = model.generate(**inputs, **gen_kwargs)
    # output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    # print(output)
    length = 0

    try:
        for response, history in model.stream_chat(tokenizer, inputText, history=[]):
            print(response[length:], flush=True, end="")
            length = len(response)
    except:
        continue
    #来到美丽的大自然，我们发现各种各样的花千奇百怪。有的颜色鲜艳亮丽,使人感觉生机勃勃；有的是红色的花瓣儿粉嫩嫩的像少女害羞的脸庞一样让人爱不释手．有的小巧玲珑; 还有的花瓣粗大看似枯黄实则暗藏玄机！
    #不同的花卉有不同的“脾气”,它们都有着属于自己的故事和人生道理.这些鲜花都是大自然中最为原始的物种,每一朵都绽放出别样的美令人陶醉、着迷!