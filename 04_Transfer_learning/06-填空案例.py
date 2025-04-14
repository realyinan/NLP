from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import time
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from tqdm import tqdm


# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("./model/bert-base-chinese")

# 加载预训练模型
bert_model = BertModel.from_pretrained("./model/bert-base-chinese")
bert_model.to(device)


def collate_fn1(data):
    sents = [value["text"] for value in data]
    # 进行编码
    inputs = bert_tokenizer.batch_encode_plus(sents, truncation=True, padding="max_length", max_length=32, return_tensors='pt')

    input_ids = inputs["input_ids"]  # [8, 32]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    # 选择每个样本的第16个位置信息
    labels = input_ids[:, 16].clone()
    # 将input_ids中第16个位置替换为[MASK]对应的id
    input_ids[:, 16] = bert_tokenizer.mask_token_id
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


def test_dataset():
    original_dataset = load_dataset("csv", data_files="./data/train.csv", split="train")
    # print(original_dataset)

    # 过滤出text长度大于32的样本
    train_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)
    # print(train_dataset)

    # 实例化dataloader对象
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1)

    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(bert_tokenizer.decode(input_ids[0]))
        print(bert_tokenizer.decode(labels[0]))
        break


# 自定义模型
class FMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, bert_tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取出bert_output第十六个位置的词张量信息
        last_hidden_state = bert_output["last_hidden_state"]  # [8, 32, 768]
        result = self.linear(last_hidden_state[:, 16])  # [8, 768] -> [8, 21138]
        return result
    

def test_model():
    original_dataset = load_dataset("csv", data_files="./data/train.csv", split="train")

    # 过滤出text长度大于32的样本
    train_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)

    # 实例化dataloader对象
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1)

    # 实例化模型
    fm_model = FMModel()

    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        result = fm_model(input_ids, attention_mask, token_type_ids)
        print(result.shape)
        break


def model2train():
    original_dataset = load_dataset("csv", data_files="./data/train.csv", split="train")

    train_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)

    # 实例化模型
    fm_model = FMModel()
    fm_model.to(device)

    # 实例化优化器函数
    fm_adamw = optim.AdamW(fm_model.parameters(), lr=5e-4)

    # 实例化损失函数
    fm_entropy = nn.CrossEntropyLoss()

    fm_model.train()

    epochs = 3

    for epoch_idx in range(epochs):
        start_time = time.time()
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1)
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_dataloader), start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = fm_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            fm_loss = fm_entropy(output, labels)
            # 梯度清零
            fm_adamw.zero_grad()
            # 后向传播
            fm_loss.backward()
            # 梯度更新
            fm_adamw.step()

            # 每5步打印日志
            if i % 5 == 0:
                temp_idx = torch.argmax(output, dim=-1)
                # 得到准确率
                acc = (temp_idx==labels).sum().item() / len(labels)
                use_time = time.time()
                print(f"当前轮次: {epoch_idx+1}, 损失: {fm_loss.detach().item():.3f}, 准确率: {acc:.3f}, 时间: {use_time-start_time:.3f}")
        torch.save(fm_model.state_dict(), f="./model/fm_model.bin")


def model2dev():
    original_dataset = load_dataset("csv", data_files="./data/test.csv", split="train")

    dev_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)

    # 实例化dataloader
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1)

    # 实例化模型
    fm_model = FMModel()
    fm_model.load_state_dict(torch.load("./model/fm_model.bin"))

    fm_model.eval()

    # 开始预测
    correct = 0
    total =0
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(dev_dataloader), start=1):
            output = fm_model(input_ids, attention_mask, token_type_ids)
            temp_idx = torch.argmax(output, dim=-1)
            correct = correct + (temp_idx==labels).sum().item()
            total = total + len(labels)

            if i % 5 == 0:
                acc = correct / total
                # 选择第一个样本进行展示
                text = bert_tokenizer.decode(input_ids[0])
                predict = bert_tokenizer.decode(temp_idx[0])
                label = bert_tokenizer.decode(labels[0])
                print(text, predict, label, acc)


if __name__ == "__main__":
    test_dataset()
    # test_model()
    # model2train()
    # model2dev()
