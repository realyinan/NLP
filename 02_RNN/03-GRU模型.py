import torch.nn as nn
import torch


def gru_use():
    # 实例化模型
    gru = nn.GRU(5, 6, 1)

    # 输入
    input = torch.randn(4, 3, 5)
    h0 = torch.randn(1, 3, 6)

    # 送入模型
    output, hn = gru(input, h0)

    print(output)
    print(hn)



if __name__ == "__main__":
    gru_use()
