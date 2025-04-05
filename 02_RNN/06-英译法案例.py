import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义开始字符
SOS_token = 0

# 定义结束字符
EOS_token = 1

# 定义句子最大长度(包含标点符号)
MAX_LENGTH = 10

# 定义数据集路径
data_path = "./data/eng-fra-v2.txt"

# 文本清洗工具
def normalizeString(s: str):
    s = s.lower().strip()
    # 再.?!前加一个空格
    s = re.sub(pattern=r"([.?!])", repl=r" \1", string=s)
    # 不是 大小写字母和正常标点的都替换成空格
    s = re.sub(pattern=r"[^a-zA-Z.!?]+", repl=r" ", string=s)
    return s


# 实现文本数据清洗+构建词典
def get_data():
    # 读取数据
    with open(data_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        # print(lines[:3])

        # 得到数据对[[英文, 法文], [英文, 发文], ...]
        my_pairs = [[normalizeString(s) for s in line.split("\t")] for line in lines]
        # print(my_pairs[:3])

        # 构建英文字典和法文字典
        english_word2index = {"SOS": 0, "EOS": 1}
        english_word_n = 2
        french_word2index = {"SOS": 0, "EOS": 1}
        french_word_n = 2

        for pair in my_pairs:
            for word in pair[0].split(" "):
                if word not in english_word2index:
                    english_word2index[word] = english_word_n
                    english_word_n += 1
                    # english_word2index[word] = len(english_word2index)
            
            for word in pair[1].split(" "):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1

        english_index2word = {v: k for k, v in english_word2index.items()}
        french_index2word = {v: k for k, v in french_word2index.items()}

    return english_word2index, english_index2word, french_word2index, french_index2word, english_word_n, french_word_n, my_pairs


english_word2index, english_index2word, french_word2index, french_index2word, english_word_n, french_word_n, my_pairs = get_data()


class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len
    
    def __getitem__(self, index):
        index = min(max(0, index), self.sample_len-1)
        # 根据索引取出对应的英文和法文
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]
        # 将x, y 张量化表示
        x1 = [english_word2index[word] for word in x.split(" ")]
        x1.append(EOS_token)
        tensor_x = torch.tensor(x1, dtype=torch.long, device=device)

        y1 = [french_word2index[word] for word in y.split(" ")]
        y1.append(EOS_token)
        tensor_y = torch.tensor(y1, dtype=torch.long, device=device)
        return tensor_x, tensor_y
    

# 定义GRU编码器模型
class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # vocab_size->代表去重之后单词的总个数
        # hidden_size->代表词嵌入维度
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # input->[1, 6]->[1, 6, 256]
        input = self.embd(input)
        # 将数据送入GRU
        # hidden->[1, 1, 256]
        output, hn = self.gru(input, hidden)  # output->[1, 6, 256], hn->[1, 1, 256]
        return output, hn
    
    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

def test_encoder():
    mydataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    my_encoder_gru = EncoderGRU(english_word_n, 256)
    my_encoder_gru = my_encoder_gru.to(device)
    for x, y in my_dataloader:
        h0 = my_encoder_gru.inithidden()
        output, hn = my_encoder_gru(x, h0)
        print(f"output: {output.shape}")
        print(f"hn: {hn.shape}")
        break


# 定义不带attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self, french_vocab_size, hidden_size):
        super().__init__()
        # 代表法文单词的总个数
        self.vocab_size = french_vocab_size
        # hidden_size代表词嵌入维度
        self.hidden_size = hidden_size
        # 定义Embeding层
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 定义softmax函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input->[1, 1] 一个字符一个字符去解码
        # 将input进行Embedding
        # input1->[1, 1, 256]
        input1 = self.embd(input)
        # 对embedding之后的结果进行relu激活, 防止过拟合
        input1 = F.relu(input1)
        # 将input1和hidden送入gru模型
        # hidden->[1, 1, 256]
        # output->[1, 1, 256]
        # hn->[1, 1, 256]
        output, hidden = self.gru(input1, hidden)
        # 对output送入输出层
        # output[0]->[1, 256]
        # result->[1, 4345]
        result = self.softmax(self.out(output[0]))
        return result, hidden
    
    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

# 测试不带attention的解码器
def test_decoder():
    mydataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    my_encoder_gru = EncoderGRU(english_word_n, 256)
    my_encoder_gru = my_encoder_gru.to(device)
    my_decoder_gru = DecoderGRU(french_word_n, 256)
    my_decoder_gru = my_decoder_gru.to(device)
    for i, (x, y) in enumerate(my_dataloader):
        # 将x送入编码器
        h0 = my_encoder_gru.inithidden()
        encoder_output, hidden = my_encoder_gru(x, h0)
        # 解码: 一个字符一个字符去解码
        for j in range(y.shape[1]):
            temp = y[0][j].view(1, -1)
            output, hidden = my_decoder_gru(temp, hidden)
            print(output.shape)
        break


# 带attntion的解码器
class AttenDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        # vocab_size: 法文单词的总个数
        self.vocab_size = vocab_size
        # hidden_size: 词嵌入的维度
        self.hidden_size = hidden_size
        # 随机失活的概率
        self.dropout_p = dropout_p
        # 最大句子长度
        self.max_length = max_length
        # 定义Embedding
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)
        # 定义第一个线性层: 得到注意力的权重分数
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        # 定义第二个全连接层: 让注意力按照指定维度输出
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义out输出层
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 实例化softmax层, 数值化归一, 方便分类
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        # input->Q[1, 1]
        # hidden->K->[1, 1, 256]
        # encoder_output->V->[max_len, 256]
        # 1. 将input经过Embedding->[1, 1, 256]
        embeded = self.embd(input)
        # 2. 将embded的结果经过dropout防止过拟合, 不改变形状
        embeded = self.dropout(embeded)
        # 3. 按照注意力计算的第一步: 将Q, K, V按照第一计算规则来计算
        # attn_weight->[1, 10]
        attn_weight = F.softmax(self.attn(torch.cat((embeded[0], hidden[0]), dim=-1)), dim=-1)
        # 3.1 将atten_weight和V相乘
        # attn_applied->[1, 1, 256]
        attn_applied = torch.bmm(attn_weight.unsqueeze(dim=0), encoder_outputs.unsqueeze(dim=0))
        # 3.2 因为第一步是拼接, 所以需要将Q和attn_applied结果再次拼接
        # cat_tensor->[1, 512]
        cat_tensor = torch.cat((embeded[0], attn_applied[0]), dim=-1)
        # 3.3 经过线性变化, 按照指定维度输出
        # atten->[1, 1, 256]
        atten = self.attn_combine(cat_tensor.unsqueeze(dim=0))
        # 4. 将注意力的结果进行relu函数激活, 防止过拟合
        atten = F.relu(atten)
        # 5. 将atten和hidden共同送入GRU模型
        # gru_output->[1, 1, 256]
        # hidden->[1, 1, 256]
        gru_output, hidden = self.gru(atten, hidden)
        # 6. 经过输出层
        # result->[1, 4345]
        result = self.softmax(self.out(gru_output[0]))
        return result, hidden, attn_weight


# 测试带attention的解码器
def test_attenDecoder():
    mydataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    encoder_gru = EncoderGRU(english_word_n, 256)
    encoder_gru = encoder_gru.to(device)

    # 实例化带attention的解码器
    attention_decoder_gru = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    attention_decoder_gru = attention_decoder_gru.to(device)

    # 迭代数据
    for i, (x, y) in enumerate(my_dataloader):
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")

        # 将x送入编码器
        h0 = encoder_gru.inithidden()
        output, hidden = encoder_gru(x, h0)
        print(f"output: {output.shape}")
        print(f"hidden: {hidden.shape}")

        # 准备解码器的输入
        encoder_c = torch.zeros(MAX_LENGTH, encoder_gru.hidden_size, device=device)
        # 将真实的编码结果赋值
        for idx in range(output.shape[1]):
            encoder_c[idx] = output[0][idx]
        print(encoder_c)

        # 解码: 一个字符一个字符解码
        for idx in range(y.shape[1]):
            temp = y[0][idx].view(1, -1)
            decoder_output, hidden, atten_weight = attention_decoder_gru(temp, hidden, encoder_c)
            print(f"decoder_output: {decoder_output.shape}")
        break



if __name__ == "__main__":
    # test_encoder()
    # test_decoder()
    test_attenDecoder()
    ...
