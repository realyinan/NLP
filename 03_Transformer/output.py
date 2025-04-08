import torch.nn as nn
import torch.nn.functional as F
from decoder import *


# 定义输出层
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # d_model 代表词嵌入维度
        # vocab_size 代表最后的输出维度: 词表的大小
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        y = F.log_softmax(self.linear(x), dim=-1)
        return y
    

def test_output():
    decoder_output = test_decoder()
    # 实例化输出层
    my_generator = Generator(512, 2000)
    result = my_generator(decoder_output)
    print(f"最终transformer结果的输出: {result.shape}")
    print(result)


if __name__ == "__main__":
    test_output()
    ...
