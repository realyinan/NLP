from input import *


# 定义下三角矩阵
def sub_mask(size):
    # 生成上三角矩阵
    up_vector = np.triu(np.ones((1, size, size)), k=1).astype("uint8")

    # 变成下三角矩阵
    return torch.from_numpy(1-up_vector)


def attention(query, key, value, mask=None, dropout = None):
    # query, key, value分别来自输入部分，如果是自注意力：query=key=value;如果不是自注意力：query!=key=value
    # query, key, value形状一致：[2, 4, 512]
    # mask代表是否对注意力分数进行掩码，如果用到编码器第一层，那就是padding——mask,防止补齐的元素对注意力产生影响
    # 如果用到解码器第一层,sentence-mask,解码的时候，防止未来信息被提前利用
    # dropout随机失活的对象，防止过拟合

    # 1. 获取d_k代表词嵌入维度
    d_k = query.size(-1)

    # 计算注意力分数
    # [2, 4, 512]*[2, 512, 4] -> [2, 4, 4]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果进行掩码, 我们的掩码张量需要作用到注意力分数上
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)

    # 对注意力分数进行softmax归一化, 得到注意力权重
    p_atten = F.softmax(scores, dim=-1)

    # 为了防止过拟合, 我们需要对p_atten进行dropout
    if dropout is not None:
        p_atten = dropout(p_atten)

    # 计算出注意力
    # [2, 4, 4] * [2, 4, 512] = [2, 4, 512]
    atten = torch.matmul(p_atten, value)

    return atten, p_atten


def test_attention():
    # 调用函数获取: 输入部分
    position_x = test_positionEncoding()
    # 准备数据
    query = key = value = position_x
    atten1, p_atten1 = attention(query, key, value)
    print("注意力结果: ", atten1.shape)
    print("注意力权重: ", p_atten1)
    mask = torch.zeros(2, 4, 4)
    atten2, p_atten2 = attention(query, key, value, mask=mask)
    print("注意力结果: ", atten2.shape)
    print("注意力权重: ", p_atten2)


# 克隆N个神经网络的某一层
def clones(model, N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


# 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embed_dim, dropout_p=0.1):
        super().__init__()
        # head 指定多少个头 embed_dim词嵌入维度 
        # 判断词嵌入维度是否可以整除
        assert embed_dim % head == 0
        # 获得每个头的词嵌入表示维度 embed_dim = 512, head = 8, d_k=64
        self.d_k = embed_dim // head
        # 指定head的属性
        self.head = head
        # 定义4个全连接层
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        # 定义一个atten的属性
        self.atten = None
        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)


def test_MultiHeadAttention():
    my_atten = MultiHeadAttention(8, 512)
    print(my_atten)

if __name__ == "__main__":
    # test_attention()
    test_MultiHeadAttention()