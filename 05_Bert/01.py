from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained("./model/bert-base-chinese")
print(model)