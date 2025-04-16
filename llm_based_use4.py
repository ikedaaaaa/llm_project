import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline


model_id = "cyberagent/open-calm-small"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


"""句点をeos_token_idに指定して5つの文章を生成する"""
# print(tokenizer.encode("。"))
input = tokenizer("東京は日本の", return_tensors="pt")

tokens = model.generate(**input, 
                        max_new_tokens=30,
                        eos_token_id = tokenizer.encode("。"),
                        pad_token_id = tokenizer.pad_token_id,
                        do_sample = True,
                        num_return_sequences= 5) 

for i in range(5):
    output = tokenizer.decode(tokens[i], skip_special_tokens=True)
    print(f"output {i+1}:",output)
    

