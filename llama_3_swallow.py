import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

    
    
model_id = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1" # 8Bモデル,量子化できない
# model_id = "tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1" # 量子化できる

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # quantization_config=BitsAndBytesConfig(load_in_4bit=True), # 量子化をする場合
    device_map="auto",
)


DEFAULT_SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。"
text = "熊本県の観光地を教えて下さい。"

messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": text},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

token_ids = tokenizer.encode(
    prompt, add_special_tokens=False,
    return_tensors="pt"
)

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

output = tokenizer.decode(
    output_ids.tolist()[0][token_ids.size(1):], 
    skip_special_tokens=True
)
print(output)