from input import *


# 定义下三角矩阵
def sub_mask(size):
    # 生成上三角矩阵
    up_vector = np.triu(np.ones((1, size, size)), k=1).astype("uint8")

    # 变成下三角矩阵
    return torch.from_numpy(1-up_vector)


# 克隆N个神经网络的某一层
def clones(model, N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


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
    atten_weight = F.softmax(scores, dim=-1)

    # 为了防止过拟合, 我们需要对p_atten进行dropout
    if dropout is not None:
        atten_weight = dropout(atten_weight)

    # 计算出注意力
    # [2, 4, 4] * [2, 4, 512] = [2, 4, 512]
    atten = torch.matmul(atten_weight, value)

    return atten, atten_weight


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

    def forward(self, query, key, value, mask=None):
        # query, key, value 都来自输入部分, 如果是自注意力, 它们相等
        # Q, K, V -> [2, 4, 512]
        # mask -> [8, 4, 4] -> [1, 8, 4, 4]
        if mask is not None:
            mask = mask.unsqueeze(dim=0)

        # 获取batch_size
        batch_size = query.size(0)

        # 获取多头注意力运算的Q, K, V
        # model(x) -> [2, 4, 512]
        # model(x).view(batch_size, -1, self.head, self.d_k) -> [2, 4, 8, 64]
        # model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) -> [2, 8, 4, 64] [seq_len, enbed_dim]
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) 
                             for model, x in zip(self.linears, (query, key, value))]
        
        # query, key, value的形状都是[2, 8, 4, 64]
        # 调用注意力
        # x -> [2, 8, 4, 64] atten_weight -> [2, 8, 4, 4]
        x, atten_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 需要对上一步计算的注意力结果进行融合
        # x -> [2, 4, 512]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)

        return self.linears[-1](x)


def test_MultiHeadAttention():
    my_atten = MultiHeadAttention(8, 512)
    position_x = test_positionEncoding()
    query = key = value = position_x
    mask = torch.zeros(8, 4, 4)
    result = my_atten(query, key, value, mask=mask)
    print(f"多头注意力机制的结果: {result.shape}")
    return result


# 定义前馈全连接层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p = 0.1):
        super().__init__()
        # d_model: 词嵌入维度
        # d_ff: 代表中间表示的维度
        # 定义第一个全连接层
        self.linear1 = nn.Linear(d_model, d_ff)
        # 定义第二个全连接层
        self.linear2 = nn.Linear(d_ff, d_model)
        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

def test_ff():
    x = test_MultiHeadAttention()
    my_ff = FeedForward(d_model=512, d_ff=1024)
    output = my_ff(x)
    print(f"前馈连接层注意力的结果: {output.shape}")
    return output


# 定义规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # features 词嵌入维度
        # 定义系数a, 会随着模型的训练, 进行参数的更改
        self.a = nn.Parameter(torch.ones(features))

        # 定义偏置 这两个参数是可学习的，意思是模型可以自动学习出适合当前任务的缩放和偏置，从而保留归一化的好处，但又不牺牲网络的表现力。
        self.b = nn.Parameter(torch.zeros(features))

        # 定义一个常量 防止分母为0
        self.eps = eps

    def forward(self, x):
        # x -> [2, 4, 512]
        # 1. 求均值
        x_mean = torch.mean(x, dim=-1, keepdim=True)

        # 2. 求标准差
        x_std = torch.std(x, dim=-1, keepdim=True)

        # 计算规范化值
        y = self.a * (x - x_mean) / (x_std+self.eps) + self.b

        return y
    

def test_layerNorm():
    # 得到前馈全连接层的结果
    x = test_ff()
    my_layernorm = LayerNorm(features=512)
    y = my_layernorm(x)
    print(y)
    print(f"规范化后的结果: {y.shape}")


# 定义子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p = 0.1):
        super().__init__()
        # size: 词嵌入维度
        # 实例化一个标准化层
        self.norm = LayerNorm(size)

        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        # x 代表原始输入, sublayer代表函数对象
        # 方式1
        result = x + self.dropout(self.norm(sublayer(x)))

        # 方式2
        # result = x + self.dropout(sublayer(self.norm(x)))

        return result


def test_sublayer():
    x = test_positionEncoding()
    mask = torch.zeros(8, 4, 4)

    self_atten = MultiHeadAttention(8, 512)
    sublayer = lambda x: self_atten(x, x, x, mask)
    sc = SublayerConnection(512)

    sc_result = sc(x, sublayer)
    print(f"子层连接结构的结果: {sc_result.shape}")
    print(sc_result)


# 定义整个编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_atten, feed_forward, dropout_p=0.1):
        super().__init__()
        # self_atten: 代表实例化后的多头注意力对象
        self.self_atten = self_atten
        # feed_forward: 代表实例化后的前馈全连接层对象
        self.feed_forward = feed_forward
        # size: 词嵌入维度
        self.size = size
        # 定义两层子层连接结构
        self.sublayers = clones(SublayerConnection(size, dropout_p), 2)

    def forward(self, x, mask):
        # 经过第一个子层连接结构
        x1 = self.sublayers[0](x, lambda x: self.self_atten(x, x, x, mask))
        # 经过第二个子层连接结构
        x2 = self.sublayers[1](x1, lambda x: self.feed_forward(x))

        return x2
    

def test_encoderlayer():
    x = test_positionEncoding()
    mask = torch.zeros(8, 4, 4)

    # 实例化多头注意力层
    head = 8
    d_model = embed_dim = 512
    self_atten = MultiHeadAttention(head, embed_dim)

    # 实例化前馈全连接层
    feed_forward = FeedForward(d_model=d_model, d_ff=1024)

    # 实例化编码器层
    my_encoderlayer = EncoderLayer(size=d_model, self_atten=self_atten, feed_forward=feed_forward)

    output = my_encoderlayer(x, mask)

    print(f"一个编码器层得到的结果: {output.shape}")


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # layer代表编码器层的对象 N代表编码器层的个数
        # 克隆N个编码器
        self.layers = clones(layer, N)

        # 定义给规范化层 作用在最后一个编码器层的输出结果上
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

def test_encoder():
    x = test_positionEncoding()
    mask = torch.zeros(8, 4, 4)

    # 实例化多头注意力层
    head = 8
    d_model = embed_dim = 512
    self_atten = MultiHeadAttention(head, embed_dim)

    # 实例化前馈全连接层
    feed_forward = FeedForward(d_model=d_model, d_ff=1024)

    # 实例化编码器层
    my_encoderlayer = EncoderLayer(size=d_model, self_atten=self_atten, feed_forward=feed_forward)

    # 实例化编码器
    my_encoder = Encoder(my_encoderlayer, 6)

    # 数据送给编码器
    output = my_encoder(x, mask)

    print(f"编码器得到的结果: {output.shape}")
    print(output)
    return output
    

if __name__ == "__main__":
    # test_attention()
    # test_MultiHeadAttention()
    # test_ff()
    # test_layerNorm()
    # test_sublayer()
    # test_encoderlayer()
    test_encoder()
    ...