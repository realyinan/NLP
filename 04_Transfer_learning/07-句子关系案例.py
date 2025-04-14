from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import time
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from tqdm import tqdm
import random


# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("./model/bert-base-chinese")

# 加载预训练模型
bert_model = BertModel.from_pretrained("./model/bert-base-chinese")
bert_model.to(device)


class Mydataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        original_dataset = load_dataset("csv", data_files=data_path, split="train")
        self.dataset = original_dataset.filter(lambda x: len(x["text"]) > 44)
        self.sample_len = len(self.dataset)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        label = 1
        text = self.dataset[index]["text"]
        sent1 = text[:22]
        sent2 = text[22:44]
        if random.randint(0, 1) == 0:
            j = random.randint(0, self.sample_len-1)
            sent2 = self.dataset[j]["text"][22:44]
            label = 0
        return sent1, sent2, label


def collate_fn1(data):
    sents = [value[:2] for value in data]
    labels = [value[-1] for value in data]

    inputs = bert_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents, padding="max_length", max_length=50, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]

    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, token_type_ids, attention_mask, labels


def test_dataset():
    my_dataset = Mydataset(data_path="./data/train.csv")

    # 使用dataloader进行再次的封装
    train_dataloader = DataLoader(dataset=my_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn1)

    for _ in train_dataloader:
        break


class NSPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层层  768代表预训练模型的输出的词嵌入维度, 2代表2分类
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将数据送入预训练模型, 得到特征向量表示, 不更新预训练模型的参数
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
 
        # 直接拿pooler_output这个结果去做下一步处理
        result = self.linear(bert_output["pooler_output"])  #  [8, 768] -> [8, 2]
        return result
    

def test_model():
    my_dataset = Mydataset(data_path="./data/train.csv")

    # 使用dataloader进行再次的封装
    train_dataloader = DataLoader(dataset=my_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn1)   

    nsp_model = NSPModel()

    for input_ids, token_type_ids, attention_mask, labels in train_dataloader:
        output = nsp_model(input_ids, token_type_ids, attention_mask)
        print(output.shape)
        break


def model2train():
    # 获取数据
    my_dataset = Mydataset(data_path="./data/train.csv")

    # 实例化模型
    nsp_model = NSPModel()
    nsp_model.to(device)

    # 实例化优化器
    nsp_adamw = optim.AdamW(nsp_model.parameters(), lr=5e-4)

    # 实例化损失函数对象
    nsp_entropy = nn.CrossEntropyLoss()

    # 对预训练模型参数不更新
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 定义当前模型为训练模式
    nsp_model.train()

    # 定义参数
    epochs = 3

    for epoch_idx in range(epochs):
        start_time = time.time()
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=my_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1, drop_last=True)
        # 开始内部迭代
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(tqdm(train_dataloader), start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)
            # 将数据送入模型得到预测结果
            output = nsp_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            nsp_loss = nsp_entropy(output, label)
            # 梯度清零
            nsp_adamw.zero_grad()
            # 反向传播
            nsp_loss.backward()
            # 梯度更新
            nsp_adamw.step()
            # 打印日志
            if i % 5 == 0:
                # 求出预测结果中[8, 2] 8个样本中最大概率值对应的索引
                temp_idx = torch.argmax(output, dim=-1)
                acc = (temp_idx == label).sum().item() / len(label)
                use_time = time.time()
                print(f"当前轮次: {epoch_idx+1}, 损失: {nsp_loss.detach().item():.3f}, 准确率: {acc:.3f}, 时间: {use_time-start_time:.3f}")
        torch.save(nsp_model.state_dict(), f="./model/nsp_model.bin")


def model2dev():
    # 加载测试数据
    my_dataset = Mydataset(data_path="./data/test.csv")
    dev_dataloader = DataLoader(dataset=my_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn1)

    # 加载训练好的模型参数
    nsp_model = NSPModel()
    nsp_model.load_state_dict(torch.load("./model/nsp_model.bin"))

    # 设置模型为评估模式
    nsp_model.eval()

    # 定义参数
    correct = 0  # 预测正确的样本个数
    total = 0  # 已经预测的样本个数

    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(dev_dataloader), start=1):
            output = nsp_model(input_ids, attention_mask, token_type_ids)
            temp_idx = torch.argmax(output, dim=-1)
            correct = correct + (temp_idx == labels).sum().item()
            total = total + len(labels)

            # 每隔五步打印平均准确率, 并随机选择一个样本查验模型效果
            if i % 5 == 0:
                acc = correct / total
                print(f"准确率: {acc}")


if __name__ == "__main__":
    # test_dataset()
    # test_model()
    # model2train()
    model2dev()
    ...