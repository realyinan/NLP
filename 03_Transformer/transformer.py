from output import *


# 定义transformer架构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        # encoder 编码器对象
        self.encoder = encoder
        # decoder 解码器对象
        self.decoder = decoder
        # src_embed 编码器的输入部分(词嵌入层+位置编码)
        self.src_embed = src_embed
        # tgt_embed 解码器的输入(词嵌入层+位置编码)
        self.tgt_embed = tgt_embed
        # generator 代表输出层对象
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        source 代表原始编码器的输入 [2, 4]
        target 代表原始解码器的输入 [2, 4]
        source_mask 代表padding mask
        target_mask 代表 scentence_mask
        """

        # 将原始的输入 source 送入(词嵌入层+位置编码器层)
        # embed_x [2, 4, 512]
        src_embed_x = self.src_embed(source) 

        # 送入编码器得到编码器的输出
        encoder_output = self.encoder(src_embed_x, source_mask)

        # 将原始的输入 target 送入(词嵌入层+位置编码器层)
        tgt_embed_x = self.tgt_embed(target)

        # 将编码器的输出结果送入解码器得到输出
        decoder_output = self.decoder(tgt_embed_x, encoder_output, source_mask, target_mask)

        # 将解码器的输出结果送入输出层得到最终的输出结果
        result = self.generator(decoder_output)

        return result
    

def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    # 实例化编码器输入词嵌入对象
    encoder_embed = MyMyEmbedding(vocab_size=source_vocab, d_model=d_model)
    # 实例化解码器输入词嵌入对象
    decoder_embed = MyMyEmbedding(vocab_size=target_vocab, d_model=d_model)
    # 实例化位置编码器对象
    position_encoder = PositionEncoding(d_model=d_model, dropout_p=dropout)
    # 实例化多头注意力对象
    atten = MultiHeadAttention(head=head, embed_dim=d_model)
    # 实例化前馈全连接层
    ff = FeedForward(d_model=d_model, d_ff=d_ff)

    c = copy.deepcopy

    # 获得编码器对象
    encoder = Encoder(layer=EncoderLayer(size=d_model, self_atten=c(atten), feed_forward=c(ff)), N=N)

    # 获得解码器对象
    decoder = Decoder(layer=DecoderLayer(size=d_model, self_atten=c(atten), src_atten=c(atten), feed_forward=c(ff)), N=N)

    # 获取编码器的输入部分
    src_embed = nn.Sequential(encoder_embed, c(position_encoder))

    # 获取解码器的输入部分
    tgt_embed = nn.Sequential(decoder_embed, c(position_encoder))

    # 获取输出部分
    generator = Generator(d_model=d_model, vocab_size=target_vocab)

    
    my_transformer = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in my_transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return my_transformer


if __name__ == "__main__":
    model = make_model(1000, 1000)
    source = torch.tensor([[1, 10, 20, 30], [3, 5, 90, 7]], dtype=torch.long)
    target = torch.tensor([[4, 100, 2, 30], [3, 51, 90, 6]], dtype=torch.long)
    source_mask = target_mask = torch.zeros(8, 4, 4)
    result = model(source, target, source_mask, target_mask)
    print(f"最终输出: {result.shape}")
    print(result)
