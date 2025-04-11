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
# bert_model.to(device)


def loaddataset():
    # 加载训练集
    # 如果没有注明split="train", 返回的结果类型是: DatasetDict样式
    train_dataset = load_dataset('csv', data_files="./data/train.csv", split="train")
    print(train_dataset)
    print(len(train_dataset))
    print(train_dataset[0])
    print(train_dataset[:3])


def collate_fn1(data):
    # data 是一个列表样式: [{'label': 0, 'text': '....'}, {}, {}]
    # 自定义函数在调用dataloader时, 自动会被使用, batch_size等于几, 自动传递几个样本(包含标签和文本)
    # 取出对应的text和标签
    seqs = [value['text'] for value in data]
    labels = [value['label'] for value in data]

    # 对一个批次的所有样本进行编码, 实现张量表示
    inputs = bert_tokenizer.batch_encode_plus(
        seqs,
        truncation=True,
        padding="max_length",
        max_length=300,
        return_tensors='pt',
    )

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    label = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_mask, token_type_ids, label


def test_dataset():
    # 加载训练集的dataset对象
    train_dataset = load_dataset("csv", data_files="./data/train.csv", split="train")

    # 实例化dataloader对象
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn1
    )

    # for循环dataloader, 查验collate_fn1函数的内部逻辑
    for input_ids, attention_mask, token_type_ids, label in train_dataloader:
        print("input_ids.shape: ", input_ids.shape)  # [4, 300]
        print("attention_mask.shape", attention_mask.shape)  # [4, 300]
        print("token_type_ids.shape", token_type_ids.shape)  # [4, 300]
        print(label)
        break


# 自定义下游任务的模型
class ClsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层层  768代表预训练模型的输出结果, 单词的词嵌入维度, 2代表2分类
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将数据送入预训练模型, 得到特征向量表示, 不更新预训练模型的参数
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(f"bert_output: {bert_output}")
        # print(f"last_hidden_state: {bert_output['last_hidden_state'].shape}")  #[8, 300, 768]
        # print(f"pooler_output: {bert_output['pooler_output'].shape}")  # [8, 768]

        # 直接拿pooler_output这个结果去做下一步处理
        result = self.linear(bert_output["pooler_output"])  # [8, 2]
        return result



def test_model():
    train_dataset = load_dataset("csv", data_files="./data/train.csv", split="train")
    # 实例化dataloader
    train_dataloader= DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1)
    # 实例化模型
    cls_model = ClsModel()
    # 遍历数据送入模型
    for input_ids, attention_mask, token_type_ids, label in train_dataloader:
        result = cls_model(input_ids, attention_mask, token_type_ids)
        print("result.shape: ", result.shape)
        break


def model2train():
    # 获取数据
    train_dataset = load_dataset('csv', data_files="./data/train.csv", split="train")

    # 实例化模型
    cls_model = ClsModel()
    cls_model.to(device)

    # 实例化优化器
    cls_adamw = optim.AdamW(cls_model.parameters(), lr=5e-4)

    # 实例化损失函数对象
    cls_entropy = nn.CrossEntropyLoss()

    # 对预训练模型参数不更新
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 定义当前模型为训练模式
    cls_model.train()

    # 定义参数
    epochs = 1

    for epoch_idx in range(epochs):
        start_time = time.time()
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn1, drop_last=True)
        # 开始内部迭代
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(tqdm(train_dataloader), start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)
            # 将数据送入模型得到预测结果
            output = cls_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            cls_loss = cls_entropy(output, label)
            # 梯度清零
            cls_adamw.zero_grad()
            # 反向传播
            cls_loss.backward()
            # 梯度更新
            cls_adamw.step()
            # 打印日志
            if i % 5 == 0:
                # 求出预测结果中[8, 2] 8个样本中最大概率值对应的索引
                temp_idx = torch.argmax(output, dim=-1)
                acc = (temp_idx == label).sum().item() / len(label)
                use_time = time.time()
                print(f"当前轮次: {epoch_idx+1}, 损失: {cls_loss.detach().item():.3f}, 准确率: {acc:.3f}, 时间: {use_time-start_time:.3f}")
        torch.save(cls_model.state_dict(), f="./model/cls_model.bin")


def model2dev():
    # 加载测试数据
    dev_dataset = load_dataset('csv', data_files="./data/test.csv", split="train")
    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn1)

    # 加载训练好的模型参数
    cls_model = ClsModel()
    cls_model.load_state_dict(torch.load("./model/cls_model.bin"))

    # 设置模型为评估模式
    cls_model.eval()

    # 定义参数
    correct = 0  # 预测正确的样本个数
    total = 0  # 已经预测的样本个数

    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(dev_dataloader), start=1):
            output = cls_model(input_ids, attention_mask, token_type_ids)
            temp_idx = torch.argmax(output, dim=-1)
            correct = correct + (temp_idx == labels).sum().item()
            total = total + len(labels)

            # 每隔五步打印平均准确率, 并随机选择一个样本查验模型效果
            if i % 5 == 0:
                acc = correct / total
                text = bert_tokenizer.decode(input_ids[0], skip_special_tokens=True)
                predict = temp_idx[0].item()
                label = labels[0].item()
                print(acc, text, predict, label)

    

if __name__ == "__main__":
    loaddataset()
    # test_dataset()
    # test_model()
    # model2train()
    # model2dev()
    ...
