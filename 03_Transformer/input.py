import os
import math
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 输入部分: wordEmbedding+PositionEncoding
class MyMyEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # vocab_size: 去重之后单词总个数
        # d_model: 词嵌入维度
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
    

# 定义位置编码器
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_length=60):
        super().__init__()
        # d_model: 词嵌入维度
        # dropout_p: 代表随机失活的概率
        # max_length: 最大句子长度
        self.dropout = nn.Dropout(p=dropout_p)

        # 定义位置编码矩阵 [60, 512]
        pe = torch.zeros(max_length, d_model)

        # 定义位置矩阵 [60, 1]
        position = torch.arange(0, max_length).unsqueeze(1)

        # 定义中间变化矩阵
        # [256, ]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000)/d_model)

        # position和div_term进行相乘 [60, 256]
        my_vec = position*div_term

        # 对PE的奇数位置进行sin函数表示, 偶数位用cos函数
        pe[:, 0::2] = torch.sin(my_vec)
        pe[:, 1::2] = torch.cos(my_vec)

        # 对pe进行升维 [1, 60, 512]
        pe = pe.unsqueeze(dim=0)

        # 将PE注册为模型的缓存区, 可以让模型加载其参数, 但是不更新
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x 代表词嵌入后的结果[1, 4, 512]
        # 返回包含位置编码的词向量
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

def test_positionEncoding():
    # 1. 得到enbedding后的结果
    my_enbed = MyMyEmbedding(vocab_size=1000, d_model=512)
    x = torch.tensor([
        [100, 2, 421, 508],
        [491, 999, 1, 221]
    ], dtype=torch.long)
    embed_x = my_enbed(x)
    # print(f"Embedding: {embed_x.shape}")

    # 对enbedding后的结果加上编码信息
    my_position = PositionEncoding(d_model=512, dropout_p=0.1)
    position_x = my_position(embed_x)
    # print(f"Embedding+position: {position_x.shape}")
    return position_x


def show_pe():
    my_position = PositionEncoding(d_model=20, dropout_p=0)
    embed_x = torch.zeros(1, 40, 20)
    position_x = my_position(embed_x)

    # 画图
    plt.figure()
    plt.plot(np.arange(40), position_x[0, :, 4:8])
    plt.legend([f"dim {i}" for i in [4, 5, 6, 7]])
    plt.show()



if __name__ == "__main__":
    test_positionEncoding()
    # show_pe()
    ...