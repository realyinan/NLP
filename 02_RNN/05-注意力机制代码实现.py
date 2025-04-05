import torch
import torch.nn as nn
import torch.nn.functional as F


# 实现注意力机制的计算
class MyAtten(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # 定义属性
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 定义一个全连接层: 得到注意力计算的权重分数
        # 因为Q和K需要拼接才送入Linear层, 因此该Linear层的输入维度: query_size+key_size
        # 输出维度是value_size1的原因是为了和value进行矩阵相乘
        self.attn = nn.Linear(self.query_size+self.key_size, self.value_size1)

        # 定义第二个全连接层: 按照注意力计算的第三步, 需要按照指定维度输出注意力结果, 线性变换
        # 该linear接受的输入是Q和第一步计算的结果拼接后的张量
        self.linear = nn.Linear(self.query_size+self.value_size2, self.output_size)

    def forward(self, Q, K, V):
        # 按照注意力计算第一个规则: Q和K先进行拼接, 经过Linear层, 再经过softmax得到权重分数
        # Q[0]->(1, 32), K[0]->(1, 32), cat之后(1, 64)
        atten_weight = F.softmax(self.attn(torch.cat((Q[0], K[0]), dim=-1)), dim=-1)

        # 需要将第一步计算的atten_weight和V矩阵计算, atten_weight->(1, 32), V->(1, 32, 64)
        # temp->(1, 1, 64) 第一步根据第一个注意力计算规则得到的结果
        temp = torch.bmm(atten_weight.unsqueeze(dim=0), V)

        # 因为第一步有拼接操作, 所以我们进行第二步: 将Q和temp再进行拼接
        # Q[0]->(1, 32), temp[0]->(1, 64), output->(1, 96)
        output = torch.cat((Q[0], temp[0]), dim=-1)

        # 需要根据计算步骤的第三步, 对拼接后的结果进行线性变换
        # result->(1, 1, 32)
        result = self.linear(output).unsqueeze(dim=0)

        return atten_weight, result
    

class MyAtten2(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # 定义属性
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 定义一个全连接层: 得到注意力计算的权重分数
        # 因为Q和K需要拼接才送入Linear层, 因此该Linear层的输入维度: query_size+key_size
        # 输出维度是value_size1的原因是为了和value进行矩阵相乘
        self.attn = nn.Linear(self.query_size+self.key_size, self.value_size1)

        # 定义第二个全连接层: 按照注意力计算的第三步, 需要按照指定维度输出注意力结果, 线性变换
        # 该linear接受的输入是Q和第一步计算的结果拼接后的张量
        self.linear = nn.Linear(self.query_size+self.value_size2, self.output_size)

    def forward(self, Q, K, V):
        # Q->(1, 1, 32)
        # K->(1, 1, 32)
        # (Q, K)->(1, 1, 64)
        # atten_weight->(1, 1, 32)
        atten_weight = F.softmax(self.attn(torch.cat((Q, K), dim=-1)), dim=-1)

        # atten_weight->(1, 1, 32)
        # V->(1, 32, 64)
        # temp->(1, 1, 64)
        temp = torch.bmm(atten_weight, V)

        # output->(1, 1, 96)
        output = torch.cat((Q, temp), dim=-1)

        # result->(1, 1, 32)
        result = self.linear(output)

        return atten_weight, result


if __name__ == "__main__":
    Q = torch.randn(1, 1, 32)
    K = torch.randn(1, 1, 32)
    V = torch.randn(1, 32, 64)  # 32-. value_size1, 64->value-size2

    my_attention = MyAtten2(query_size=32, key_size=32, value_size1=32, value_size2=64, output_size=32)
    atten_weight, result = my_attention(Q, K, V)
    print(atten_weight.shape)
    print(result.shape)