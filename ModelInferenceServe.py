from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

#加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("model_file/pretrained_model/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("model_file/pretrained_model/Qwen1.5-0.5B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
model.to(device)

def generate_response(input_text:str) -> str:
    inputs = tokenizer(input_text,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        max_new_tokens = 100
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True) #skip_special_tokens,这些符号不是人类语言的一部分,而是模型内部使用的控制符。
    return response

