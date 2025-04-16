"""
python3
でインタラクティブモードを起動し以下を実行してmodel,tokenierを取得する
with open('llm_based_use.py') as f:
    exec(f.read())
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small")

tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")

"""「東京は日本の」に続く1単語を推測する"""
# input = tokenizer("東京は日本の", return_tensors="pt")
# tokens = model.generate(**input, max_new_tokens=1,do_sample=False)
# tokenizer.decode(tokens[0][-1])
# 出力：首都

"""キーワード引数をtrueにした場合"""
# out =  model.generate(**input, max_new_tokens=1, return_dict_in_generate=True,output_scores=True)
# out.scores[0].shape
# 出力：torch.Size([1, 52096])

"""torch.topkを使って上位5つのindexを取得し，tikenizer.decodeで文字列に変換"""
# top5 = torch.topk(out.scores[0][0],5)
# for i in range(5):
#     print(i+1,tokenizer.decode(top5.indices[i]),top5.values[i].item())
# 出力    1 首都 18.585956573486328
#        2 未来 16.709091186523438
#        3 最 16.689172744750977
#        4 文化 16.54909324645996
#        5 「 16.422061920166016


"""追加する単語数を最大10に設定した場合"""
# input = tokenizer("日本の首都はどこですか？", return_tensors="pt")
# tokens = model.generate(**input, max_new_tokens=10, do_sample=False)
# tokenizer.decode(tokens[0], skip_special_tokens=True)
# 出力：'日本の首都はどこですか?\n「東京」という都市が、なぜ「'

"""双方の発話全体を途中までの文字列として考える"""
# input = tokenizer("今日は天気が良いですね\n"+
#                   "そうですね\n"+
#                   "どこかへ行きましょうか．",
#                    return_tensors="pt")
# tokens = model.generate(**input, max_new_tokens=20, do_sample=False)
# tokenizer.decode(tokens[0], skip_special_tokens=True)
# 出力：'今日は天気が良いですね\nそうですね\nどこかへ行きましょうか\nさて、\n今日は\n「お庭で楽しむガーデニング」\nについて\nお話したいと思います。\n'



