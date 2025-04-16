import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline


model_id = "cyberagent/open-calm-small"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


"""pipelineを使った場合"""
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

outs = generator("東京は日本の", max_length=30)

print(outs[0])