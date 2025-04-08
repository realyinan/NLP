from input import *
from encoder import *


class DecoderLayer(nn.Module):
    def __init__(self, size, self_atten, src_atten, feed_forward):
        super().__init__()
        self.size = size
        # Q=K=V
        self.self_atten = self_atten
        # Q!=K=V
        self.src_attn = src_atten
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size=size), 3)

    def forward(self, y, memory, source_mask, target_mask):
        # y 代表来自解码器的输入
        # memory 代表编码器输出结果
        # source_mask 代表解码器第二个子层, 进行注意力计算的掩码: padding_mask
        # target_mask 代表解码器第一个子层, 进行自注意力计算的掩码: sentences mask
        # 数据经过第一个子层连接结构
        y1 = self.sublayers[0](y, lambda x: self.self_atten(x, x, x, target_mask))
        # 数据经过第二个子层连接结构
        y2 = self.sublayers[1](y1, lambda x: self.src_attn(x, memory, memory, source_mask))
        # 数据经过第三个子层连接结构
        y3 = self.sublayers[2](y2, lambda x: self.feed_forward(x))

        return y3
    

def test_decoderlayer():
    # 实例化多头注意力对象
    head = 8
    embed_dim = 512
    self_atten = MultiHeadAttention(head=head, embed_dim=embed_dim)
    src_atten = MultiHeadAttention(head=head, embed_dim=embed_dim)

    # 实例化前馈全连接层对象
    d_model = 512
    d_ff = 1024
    feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    # 实例化解码器层对象
    size = 512
    my_decoderlayer = DecoderLayer(size=size, self_atten=self_atten, src_atten=src_atten, feed_forward=feed_forward)

    # 获得解码器输入
    y = torch.tensor([
        [20, 40, 1, 5],
        [8, 90, 18, 24]
    ], dtype=torch.long)

    my_embed = MyMyEmbedding(vocab_size=2000, d_model=d_model)
    embed_y = my_embed(y)
    my_position = PositionEncoding(d_model=d_model, dropout_p=0.1)
    position_y = my_position(embed_y)

    # 获得编码器的输入
    encoder_output = test_encoder()

    # 获得掩码
    source_mask = target_mask = torch.zeros(8, 4, 4)

    output = my_decoderlayer(position_y, encoder_output, source_mask, target_mask)
    print(f"解码器层的输出: {output.shape}")
    print(output)


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # 定义N个解码器层
        self.layers = clones(layer, N)

        # 定义规范化层
        self.norm = LayerNorm(features=layer.size)

    def forward(self, y, memory, source_mask, taget_mask):
        for layer in self.layers:
            y = layer(y, memory, source_mask, taget_mask)
        return self.norm(y)
    

def test_decoder():
    head = 8
    embed_dim = 512
    d_model = 512
    d_ff = 1024
    size = 512
    # 实例化多头注意力对象
    self_atten = MultiHeadAttention(head=head, embed_dim=embed_dim)
    src_atten = MultiHeadAttention(head=head, embed_dim=embed_dim)

    # 实例化前馈全连接层对象
    feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    # 实例化解码器层对象
    my_decoderlayer = DecoderLayer(size=size, self_atten=self_atten, src_atten=src_atten, feed_forward=feed_forward)

    # 获得解码器输入
    y = torch.tensor([
        [20, 40, 1, 5],
        [8, 90, 18, 24]
    ], dtype=torch.long)

    my_embed = MyMyEmbedding(vocab_size=2000, d_model=d_model)
    embed_y = my_embed(y)
    my_position = PositionEncoding(d_model=d_model, dropout_p=0.1)
    position_y = my_position(embed_y)

    # 获得编码器的输入
    encoder_output = test_encoder()

    # 获得掩码
    source_mask = target_mask = torch.zeros(8, 4, 4)

    # 实例化解码器对象
    my_decoder = Decoder(layer=my_decoderlayer, N=6)
    decoder_output = my_decoder(position_y, encoder_output, source_mask, target_mask)

    print(f"解码器的输出: {decoder_output.shape}")
    print(decoder_output)
    return decoder_output




if __name__ == "__main__":
    # test_decoderlayer()
    test_decoder()
    ...
