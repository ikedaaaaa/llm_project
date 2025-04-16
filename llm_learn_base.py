import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""「私は犬が好き。」に対するtokenizerの出力"""
input = tokenizer.encode("私は犬が好き。", return_tensors="pt")
print(input)
a = [tokenizer.decode(input[0][i]) for i in range(len(input[0]))]
print(a)

"""モデルの出力"""
# output = model(input)
# type(output)
# print(output.logits)
# print(output.logits.shape)

"""各tokenと入力文に対する損失値の出力"""
# loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
# loss0 = loss_fn(output.logits[0],
#                  torch.tensor([3807,9439,247,-100]))
# print(loss0)

# print(torch.sum(loss0)/3)

"""入力文に対する損失値の出力"""
# loss_fn = torch.nn.CrossEntropyLoss()
# loss1 = loss_fn(output.logits[0],torch.tensor([3807,9439,247,-100]))
# print(loss1)

"""モデルから求めた損失値の出力"""
output = model(input, labels=input)
loss = output.loss
print(loss)

"""全体の損失からパラメータを更新する"""
optimizer.zero_grad()
loss.backward()
optimizer.step()
