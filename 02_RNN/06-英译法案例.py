import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 定义开始字符
SOS_token = 0

# 定义结束字符
EOS_token = 1

# 定义句子最大长度(包含标点符号)
MAX_LENGTH = 10

# 定义数据集路径
data_path = "./data/eng-fra-v2.txt"

# 文本清洗工具
def normalizeString(s: str):
    s = s.lower().strip()
    # 再.?!前加一个空格
    s = re.sub(pattern=r"([.?!])", repl=r" \1", string=s)
    # 不是 大小写字母和正常标点的都替换成空格
    s = re.sub(pattern=r"[^a-zA-Z.!?]+", repl=r" ", string=s)
    return s


# 实现文本数据清洗+构建词典
def get_data():
    # 读取数据
    with open(data_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        # print(lines[:3])

        # 得到数据对[[英文, 法文], [英文, 发文], ...]
        my_pairs = [[normalizeString(s) for s in line.split("\t")] for line in lines]
        # print(my_pairs[:3])

        # 构建英文字典和法文字典
        english_word2index = {"SOS": 0, "EOS": 1}
        english_word_n = 2
        french_word2index = {"SOS": 0, "EOS": 1}
        french_word_n = 2

        for pair in my_pairs:
            for word in pair[0].split(" "):
                if word not in english_word2index:
                    english_word2index[word] = english_word_n
                    english_word_n += 1
                    # english_word2index[word] = len(english_word2index)
            
            for word in pair[1].split(" "):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1

        english_index2word = {v: k for k, v in english_word2index.items()}
        french_index2word = {v: k for k, v in french_word2index.items()}

    return english_word2index, english_index2word, french_word2index, french_index2word, english_word_n, french_word_n, my_pairs


english_word2index, english_index2word, french_word2index, french_index2word, english_word_n, french_word_n, my_pairs = get_data()


class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len
    
    def __getitem__(self, index):
        index = min(max(0, index), self.sample_len-1)
        # 根据索引取出对应的英文和法文
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]
        # 将x, y 张量化表示
        x1 = [english_word2index[word] for word in x.split(" ")]
        x1.append(EOS_token)
        tensor_x = torch.tensor(x1, dtype=torch.long)

        y1 = [french_word2index[word] for word in y.split(" ")]
        y1.append(EOS_token)
        tensor_y = torch.tensor(y1, dtype=torch.long)
        return tensor_x, tensor_y
    

# 定义GRU编码器模型
class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # vocab_size->代表去重之后单词的总个数
        # hidden_size->代表词嵌入维度
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # input->[1, 6]->[1, 6, 256]
        input = self.embd(input)
        # 将数据送入GRU
        # hidden->[1, 1, 256]
        output, hn = self.gru(input, hidden)  # output->[1, 6, 256], hn->[1, 1, 256]
        return output, hn
    
    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    

def test_encoder():
    mydataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    my_encoder_gru = EncoderGRU(english_word_n, 256)
    for x, y in my_dataloader:
        h0 = my_encoder_gru.inithidden()
        output, hn = my_encoder_gru(x, h0)
        print(f"output: {output.shape}")
        print(f"hn: {hn.shape}")
        break


# 定义不带attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self, french_vocab_size, hidden_size):
        super().__init__()
        # 代表法文单词的总个数
        self.vocab_size = french_vocab_size
        # hidden_size代表词嵌入维度
        self.hidden_size = hidden_size
        # 定义Embeding层
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 定义softmax函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input->[1, 1] 一个字符一个字符去解码
        # 将input进行Embedding
        # input1->[1, 1, 256]
        input1 = self.embd(input)
        # 对embedding之后的结果进行relu激活, 防止过拟合
        input1 = F.relu(input1)
        # 将input1和hidden送入gru模型
        # hidden->[1, 1, 256]
        # output->[1, 1, 256]
        # hn->[1, 1, 256]
        output, hidden = self.gru(input1, hidden)
        # 对output送入输出层
        # output[0]->[1, 256]
        # result->[1, 4345]
        result = self.softmax(self.out(output[0]))
        return result, hidden
    
    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

# 测试不带attention的解码器
def test_decoder():
    mydataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    my_encoder_gru = EncoderGRU(english_word_n, 256)
    my_encoder_gru = my_encoder_gru.to(device)
    my_decoder_gru = DecoderGRU(french_word_n, 256)
    my_decoder_gru = my_decoder_gru.to(device)
    for i, (x, y) in enumerate(my_dataloader):
        # 将x送入编码器
        h0 = my_encoder_gru.inithidden()
        encoder_output, hidden = my_encoder_gru(x, h0)
        # 解码: 一个字符一个字符去解码
        for j in range(y.shape[1]):
            temp = y[0][j].view(1, -1)
            output, hidden = my_decoder_gru(temp, hidden)
            print(output.shape)
        break


# 带attntion的解码器
class AttenDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        # vocab_size: 法文单词的总个数
        self.vocab_size = vocab_size
        # hidden_size: 词嵌入的维度
        self.hidden_size = hidden_size
        # 随机失活的概率
        self.dropout_p = dropout_p
        # 最大句子长度
        self.max_length = max_length
        # 定义Embedding
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)
        # 定义第一个线性层: 得到注意力的权重分数
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        # 定义第二个全连接层: 让注意力按照指定维度输出
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义out输出层
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 实例化softmax层, 数值化归一, 方便分类
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        # input->Q[1, 1]
        # hidden->K->[1, 1, 256]
        # encoder_output->V->[max_len, 256]
        # 1. 将input经过Embedding->[1, 1, 256]
        embeded = self.embd(input)
        # 2. 将embded的结果经过dropout防止过拟合, 不改变形状
        embeded = self.dropout(embeded)
        # 3. 按照注意力计算的第一步: 将Q, K, V按照第一计算规则来计算
        # attn_weight->[1, 10]
        attn_weight = F.softmax(self.attn(torch.cat((embeded[0], hidden[0]), dim=-1)), dim=-1)
        # 3.1 将atten_weight和V相乘
        # attn_applied->[1, 1, 256]
        attn_applied = torch.bmm(attn_weight.unsqueeze(dim=0), encoder_outputs.unsqueeze(dim=0))
        # 3.2 因为第一步是拼接, 所以需要将Q和attn_applied结果再次拼接
        # cat_tensor->[1, 512]
        cat_tensor = torch.cat((embeded[0], attn_applied[0]), dim=-1)
        # 3.3 经过线性变化, 按照指定维度输出
        # atten->[1, 1, 256]
        atten = self.attn_combine(cat_tensor.unsqueeze(dim=0))
        # 4. 将注意力的结果进行relu函数激活, 防止过拟合
        atten = F.relu(atten)
        # 5. 将atten和hidden共同送入GRU模型
        # gru_output->[1, 1, 256]
        # hidden->[1, 1, 256]
        gru_output, hidden = self.gru(atten, hidden)
        # 6. 经过输出层
        # result->[1, 4345]
        result = self.softmax(self.out(gru_output[0]))
        return result, hidden, attn_weight


# 测试带attention的解码器
def test_attenDecoder():
    mydataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    encoder_gru = EncoderGRU(english_word_n, 256)

    # 实例化带attention的解码器
    attention_decoder_gru = AttenDecoder(vocab_size=french_word_n, hidden_size=256)

    # 迭代数据
    for i, (x, y) in enumerate(my_dataloader):
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")

        # 将x送入编码器
        h0 = encoder_gru.inithidden()
        output, hidden = encoder_gru(x, h0)
        print(f"output: {output.shape}")
        print(f"hidden: {hidden.shape}")

        # 准备解码器的输入
        encoder_c = torch.zeros(MAX_LENGTH, encoder_gru.hidden_size)
        # 将真实的编码结果赋值
        for idx in range(output.shape[1]):
            encoder_c[idx] = output[0][idx]
        print(encoder_c)

        # 解码: 一个字符一个字符解码
        for idx in range(y.shape[1]):
            temp = y[0][idx].view(1, -1)
            decoder_output, hidden, atten_weight = attention_decoder_gru(temp, hidden, encoder_c)
            print(f"decoder_output: {decoder_output.shape}")
        break


# 定义内部迭代函数
def Train_iter(x, y, encoder, decoder, encoder_adam, decoder_adam, cross_entropy):
    # 将x 英文送入编码器, 得到编码结果
    # [1, 6], [1, 1, 256] -> [1, 6, 256], [1, 1, 256]
    encoder_output, encoder_hidden = encoder(x, encoder.inithidden())
    
    # 解码参数准备和解码
    # 统一句子长度得到V值
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for i in range(x.shape[1]):
        encoder_output_c[i] = encoder_output[0, i]

    # K值
    decoder_hidden = encoder_hidden

    # Q值
    input_y = torch.tensor([[SOS_token]], device=device)

    my_loss = 0.0
    y_len = y.shape[1]
    use_teacher_forcing = True if random.random() < teacher_forcing_radio else False

    if use_teacher_forcing:
        for idx in range(y_len):
            output_y, decoder_hidden, atten_weight = decoder(input_y, decoder_hidden, encoder_output_c)
            # 计算损失函数
            target_y = y[0][idx].view(1)
            my_loss = my_loss + cross_entropy(output_y, target_y)
            # 当前时间步的真实target进一步预测下一个单词
            input_y = y[0][idx].view(1, -1)
    else:
        for idx in range(y_len):
            output_y, decoder_hidden, atten_weight = decoder(input_y, decoder_hidden, encoder_output_c)
        # 计算损失函数
            target_y = y[0][idx].view(1)
            my_loss = my_loss + cross_entropy(output_y, target_y)

            # 获取下一个input_y 不一样
            topv, topi = torch.topk(output_y, k=1)
            if topi.item() == EOS_token:
                break
            input_y = topi.detach()

    encoder_adam.zero_grad()
    decoder_adam.zero_grad()

    my_loss.backward()

    encoder_adam.step()
    decoder_adam.step()

    return my_loss.detach().item() / y_len
        


my_lr = 1e-4
epochs = 2
teacher_forcing_radio = 0.5
print_interval_num = 1000
plot_interval_num = 100

def Train_seq2seq():
    # 获取数据
    my_dataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

    # 实例化模型
    encoder_gru = EncoderGRU(english_word_n, 256).to(device)
    atten_decoder_gru = AttenDecoder(french_word_n, 256).to(device)

    # 实例化优化器
    encoder_adam = optim.Adam(encoder_gru.parameters(), lr=my_lr)
    decoder_adam = optim.Adam(atten_decoder_gru.parameters(), lr=my_lr)

    # 实例化损失函数对象
    cross_entropy = nn.NLLLoss()

    # 定义训练日志的参数
    plot_loss_list = []

    # 开始外部训练
    for epoch_idx in range(1, epochs+1):
        # 初始化变量
        print_loss_total, plot_loss_total = 0.0, 0.0
        start_time = time.time()
        for i, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            my_loss = Train_iter(x, y, encoder=encoder_gru, decoder=atten_decoder_gru, encoder_adam=encoder_adam, decoder_adam=decoder_adam, cross_entropy=cross_entropy)

            print_loss_total += my_loss
            plot_loss_total += my_loss

            # 每隔1000步打印一下日志
            if i % 1000 == 0:
                avg_print_loss = print_loss_total / 1000
                print_loss_total = 0.0
                use_time = time.time()
                print(f"当前训练的轮次: {epoch_idx}, 损失值: {avg_print_loss:.4f}, 时间: {use_time-start_time}")
            
            # 每隔100步保存平均损失
            if i % 100 == 0:
                avg_plot_loss = plot_loss_total / 100
                plot_loss_list.append(avg_plot_loss)
                plot_loss_total = 0.0
        
        torch.save(encoder_gru.state_dict(), f=f"./model/eng2fre_encoder_{epoch_idx}.bin")
        torch.save(atten_decoder_gru.state_dict(), f=f"./model/eng2fre_decoder_{epoch_idx}.bin")
    
    # 画图展示
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig("./image/eng2fre_loss.png")
    plt.show()

    return plot_loss_list

# 预测函数
def seq2seq_Evaluate(tensor_x, my_encoder, my_decoder):
    with torch.no_grad():
        # 将x送入编码器, 得到编码之后的结果
        h0 = my_encoder.inithidden()
        encoder_output, encoder_hidden = my_encoder(tensor_x, h0)

        # 准备解码器的参数V
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)
        for idx in range(tensor_x.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]

        # K
        decoder_hidden = encoder_hidden

        # Q
        input_y = torch.tensor([[SOS_token]], device=device)

        # 准备空列表: 存储解码出来的法文单词
        decoder_list = []

        # 准备全零的二维矩阵: 存储每一个时间步得到的注意力矩阵
        decoder_attention = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        # 开始解码
        for idx in range(MAX_LENGTH):
            output_y, decoder_hidden, atten_weight = my_decoder(input_y, decoder_hidden, encoder_output_c)
         
            # 取出output_y中4345个概率值对应最大的值以及索引
            _, topi = torch.topk(output_y, k=1)
            decoder_attention[idx] = atten_weight

            # 判断中间的预测结果是否是终止符
            if topi.item() == EOS_token:
                decoder_list.append("<EOS>")
                break
            else:
                decoder_list.append(french_index2word[topi.item()])

            input_y = topi.detach()
            
        return decoder_list, decoder_attention[:idx+1]
        

def test_seq2seqEvaluate():
    # 加载训练好的模型
    my_encoder = EncoderGRU(vocab_size=english_word_n, hidden_size=256)
    my_encoder.load_state_dict(torch.load("./model/eng2fre_encoder_2.bin"))
    my_encoder.to(device=device)
    my_decoder = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    my_decoder.load_state_dict(torch.load("./model/eng2fre_decoder_2.bin"))
    my_decoder.to(device=device)

    # 准备预测数据
    my_samplepairs = [
      ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
      ['i m more than a friend .', 'je suis plus qu une amie .'],
      ['she is beautiful like her mother .', 'elle est belle comme sa mere .']
    ]

    # 循环数据, 对每一个数据进行预测, 对比真实结果
    for i, pair in enumerate(my_samplepairs):
        x = pair[0]
        y = pair[1]

        # 对x进行处理
        temp_x = [english_word2index[word] for word in x.split(" ")]
        temp_x.append(EOS_token)
        tensor_x = torch.tensor([temp_x], dtype=torch.long, device=device)
        decoder_list, atten_weight = seq2seq_Evaluate(tensor_x, my_encoder, my_decoder)
        predict = " ".join(decoder_list)
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"predict: {predict}")
        print("*"*80)


# 实现注意力权重的绘图
def test_attention():
    my_encoder = EncoderGRU(vocab_size=english_word_n, hidden_size=256)
    my_encoder.load_state_dict(torch.load("./model/eng2fre_encoder_1.bin"))
    my_encoder.to(device=device)

    my_decoder = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    my_decoder.load_state_dict(torch.load("./model/eng2fre_decoder_1.bin"))
    my_decoder.to(device=device)

    # 准备数据
    scentence = "we are the teachers ."
    temp_x = [english_word2index[word] for word in scentence.split(" ")]
    temp_x.append(EOS_token)
    tensor_x = torch.tensor([temp_x], dtype=torch.long, device=device)
    decoder_list, atten_weight = seq2seq_Evaluate(tensor_x, my_encoder, my_decoder)
    predict = " ".join(decoder_list)

    plt.matshow(atten_weight)
    plt.show()


if __name__ == "__main__":
    test_encoder()
    # test_decoder()
    # test_attenDecoder()
    # Train_seq2seq()
    # test_seq2seqEvaluate()
    # test_attention()
    ...