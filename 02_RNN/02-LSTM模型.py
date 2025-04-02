import torch.nn as nn
import torch

def lstm_use():
    # 实例化对象
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, batch_first=True)

    # 输入
    input = torch.randn(4, 4, 5)
    h0 = torch.randn(1, 4, 6)
    c0 = torch.randn(1, 4, 6)

    # 将数据送入模型
    output, (hn, cn) = lstm(input, (h0, c0))
    print("output: \n", output)
    print("hn: \n", hn)
    print("cn: \n", cn)



if __name__ == "__main__":
    lstm_use()










