import torch.nn as nn
import torch


def rnn_base():
    # 实例化RNN模型
    '''
    第一个参数: input_size(输入张量x的维度)
    第二个参数: hidden_size(隐藏层的维度， 隐藏层的神经元个数)
    第三个参数: num_layer(隐藏层的数量)
    '''
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1, batch_first=False)

    # 指定input输出
    '''
    第一个参数: sequence_length(输入序列的长度)
    第二个参数: batch_size(批次的样本数量)
    第三个参数: input_size(输入张量的维度)
    '''
    input = torch.randn(1, 3, 5)

    # 指定输入h0
    '''
    第一个参数: num_layer * num_directions(层数*网络方向)
    第二个参数: batch_size(批次的样本数)
    第三个参数: hidden_size(隐藏层的维度， 隐藏层神经元的个数)
    '''
    h0 = torch.randn(1, 3, 6)

    # 将input, h0 送入模型
    # output: 每一个样本的所有单词对应的隐藏层输出结果
    # hn: 每一个样本最后一个单词对应的隐藏层输出结果
    output, hn = rnn(input, h0)

    print(output)
    print(hn)


def rnn_sequence():
    # 实例化RNN模型
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)

    # 指定input输出
    input = torch.randn(4, 3, 5)

    # 指定输入h0
    h0 = torch.randn(1, 3, 6)

    # 将input, h0 送入模型
    output, hn = rnn(input, h0)

    print(output)
    print(hn)


# RNN模型循环机制的理解：1个样本
def rnn_for_multinum():
    # 构建模型
    rnn = nn.RNN(5, 6, 1)
    # 定义输入
    input = torch.randn(4, 1, 5)
    hidden = torch.zeros(1, 1, 6)
    # 方式1：1个1个的送字符
    # for i in range(4):
    #     tmp = input[i][0]
    #     print("tmp.shape--->", tmp.shape) # 拿出1个数组
    #     output, hidden = rnn(tmp.unsqueeze(0).unsqueeze(0), hidden)
    #     print(i+1,'output--->', output, )
    #     print(i+1,'hidden--->', hidden, )
    #     print('*'*80)
    # 一次性将数据送入模型
    hidden = torch.zeros(1, 1, 6)
    output, hn = rnn(input, hidden)
    print('output2--->', output, output.shape)
    print('hn--->', hn, hn.shape)


# RNN模型循环机制的理解:多个样本
def dm04_rnn_for_multinum():
    # 构建模型
    rnn = nn.RNN(5, 6, 1)
    # 定义输入
    input = torch.randn(4, 3, 5)
    hidden = torch.zeros(1, 3, 6)
    # 方式1：1个1个的送字符
    for i in range(4):
        tmp = input[i, :, :]
        print("tmp.shape--->", tmp.shape) # 拿出1个数组
        output, hidden = rnn(tmp.unsqueeze(0), hidden)
        print(i+1,'output--->', output, )
        print(i+1,'hidden--->', hidden, )
        print('*'*80)
    # # 一次性将数据送入模型
    # hidden = torch.zeros(1, 3, 6)
    # output, hn = rnn(input, hidden)
    # print('output2--->', output, output.shape)
    # print('hn--->', hn, hn.shape)


# RNN模型 多个隐藏层
def rnn_numlayers():
    # 实例化RNN模型
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=2)

    # 指定input输出
    input = torch.randn(4, 3, 5)

    # 指定输入h0
    h0 = torch.randn(2, 3, 6)

    # 将input, h0 送入模型
    output, hn = rnn(input, h0)

    print(output)
    print(hn)


# 改变batch_first参数
def rnn_batch_first():
    # 实例化RNN模型
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1, batch_first=True)

    # 指定input输出
    input = torch.randn(4, 3, 5)

    # 指定输入h0
    h0 = torch.randn(1, 4, 6)

    # 将input, h0 送入模型
    output, hn = rnn(input, h0)

    print(output)
    print(hn)

if __name__ == "__main__":
    # rnn_base()
    # rnn_sequence()
    # rnn_for_multinum()
    # dm04_rnn_for_multinum()
    rnn_numlayers()
    # rnn_batch_first()
