import math
import torch
import torch.nn as nn

# 输入部分: wordEmbedding+PositionEncoding
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # vocab_size: 去重之后单词总个数
        # d_model: 词嵌入维度
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
    



