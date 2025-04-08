import torch
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data import Dataset, DataLoader
import string
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json


# 获取常用的字符
all_letters = string.ascii_letters + " ,.;'"
# print(len(all_letters))
# print("n_letters: ", all_letters)


# 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
categorynum = len(categorys)
# print('categorys: ', categorys)


# 读数据到内存
def read_data(filename):
    my_list_x = []
    my_list_y = []
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) < 5:
                continue
            (x, y) = line.strip().split("\t")
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x, my_list_y


class NameClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.sample_len = len(my_list_x)

    # 可以方便直接对类的对象进行操作
    def __len__(self):
        return self.sample_len
    
    # 方便对对象进行索引操作
    def __getitem__(self, index):
        # 对异常索引进行处理
        index = min(max(index, 0), self.sample_len-1)

        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # 文本张量化
        tensor_x = torch.zeros(len(x), len(all_letters))
        for idx, letter in enumerate(x):
            tensor_x[idx][all_letters.find(letter)] = 1

        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)

        return tensor_x, tensor_y
            

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # 代表输入x的词嵌入维度
        self.input_size = input_size
        # 代表RNN的输出维度
        self.hidden_size = hidden_size
        # 代表国家类别总数
        self.output_size = output_size
        # 代表RNN层数
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    
    def forward(self, input, hidden):
        # input 输入是二维的 [seq_len, input_size] -> [6, 57]
        # hidden 三维的 [num_layers , batch_size, hidden_size] [1, 1, 128]
        # 对input进行升维  [6, 1, 57]
        input = input.unsqueeze(dim=1)

        # 将input, hidden送入模型
        # output -> [6, 1, 128]
        # hn -> [1, 1, 128]
        output, hn = self.rnn(input, hidden)

        # 取出最后一个字母对应的向量当作整个样本的向量语义表示 [1, 128]
        temp = output[-1]

        # 将temp送入全连接层  [1, 18]
        result = self.linear(temp)

        return self.softmax(result), hn
    
    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # 代表输入x的词嵌入维度
        self.input_size = input_size
        # 代表RNN的输出维度
        self.hidden_size = hidden_size
        # 代表国家类别总数
        self.output_size = output_size
        # 代表RNN层数
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    
    def forward(self, input, hidden, c):
        # input 输入是二维的 [seq_len, input_size] -> [6, 57]
        # hidden 三维的 [num_layers , batch_size, hidden_size] [1, 1, 128]
        # 对input进行升维  [6, 1, 57]
        input = input.unsqueeze(dim=1)

        # 将input, hidden送入模型
        # output -> [6, 1, 128]
        # hn. c -> [1, 1, 128]
        output, (hn, cn) = self.lstm(input, (hidden, c))

        # 取出最后一个字母对应的向量当作整个样本的向量语义表示 [1, 128]
        temp = output[-1]

        # 将temp送入全连接层  [1, 18]
        result = self.linear(temp)

        return self.softmax(result), hn, cn
    
    def inithidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c0


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # 代表输入x的词嵌入维度
        self.input_size = input_size
        # 代表RNN的输出维度
        self.hidden_size = hidden_size
        # 代表国家类别总数
        self.output_size = output_size
        # 代表RNN层数
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    
    def forward(self, input, hidden):
        # input 输入是二维的 [seq_len, input_size] -> [6, 57]
        # hidden 三维的 [num_layers , batch_size, hidden_size] [1, 1, 128]
        # 对input进行升维  [6, 1, 57]
        input = input.unsqueeze(dim=1)

        # 将input, hidden送入模型
        # output -> [6, 1, 128]
        # hn -> [1, 1, 128]
        output, hn = self.gru(input, hidden)

        # 取出最后一个字母对应的向量当作整个样本的向量语义表示 [1, 128]
        temp = output[-1]

        # 将temp送入全连接层  [1, 18]
        result = self.linear(temp)

        return self.softmax(result), hn
    
    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
    

my_lr = 1e-3
epochs = 1

# 定义RNN模型的训练函数
def train_rnn():
    my_list_x, my_list_y = read_data("./data/name_classfication.txt")
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    input_size = len(all_letters)
    hidden_size = 128
    output_size = categorynum

    my_rnn = MyRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    my_crossentrophy = nn.NLLLoss()
    my_adam = optim.Adam(my_rnn.parameters(), lr=my_lr)
    # print(my_rnn)

    # 定义训练日志的参数
    start_time = time.time()  # 开始时间
    total_iter_num = 0  # 已经训练的样本数量
    total_loss = 0.0  # 已经训练的样本总损失
    total_loss_list = []  # 每迭代100个样本, 计算平均损失并保存列表中
    total_acc_num = 0  # 已经训练的样本中预测正确的样本数量
    total_acc_list = []  # 每迭代100个样本, 计算平均准确率并保存列表中

    for epoch_idx in range(epochs):
        # 实例化Dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

        # 开始内部迭代数据
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # x -> [1, 6, 57]
            h0 = my_rnn.inithidden()  # h0 = [1, 1, 128]
            output, hn = my_rnn(x[0], h0)  # output -> [1, 18]

            # 计算损失
            my_loss = my_crossentrophy(output, y)

            # 梯度清零
            my_adam.zero_grad()

            # 反向传播
            my_loss.backward()

            # 参数更新
            my_adam.step()
            
            total_iter_num += 1
            total_loss += my_loss.item()
            # 计算已经训练的样本预测正确的个数
            tag = (1 if torch.argmax(output).item() == y.item() else 0)
            total_acc_num += tag

            # 每隔100次计算一下平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                # 平均准确率
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 每隔两千步打印一下训练日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                use_time = time.time()
                print('轮次:%d, 损失:%.6f, 时间:%d, 准确率:%.4f' %(epoch_idx+1, temp_loss, use_time - start_time, temp_acc))
        
        # 每一轮都保存模型
        torch.save(my_rnn.state_dict(), f="./model/rnn_%d.bin" % (epoch_idx+1))
    
    # 计算总时间
    all_time = time.time() - start_time

    dict1 = {
        "total_loss_list": total_loss_list,
        "all_time": all_time,
        "total_acc_list": total_acc_list
    }
    with open("./data/rnn.json", mode="w", encoding='utf-8') as f:
        f.write(json.dumps(dict1))
    return total_loss_list, all_time, total_acc_list


def train_lstm():
    my_list_x, my_list_y = read_data("./data/name_classfication.txt")
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    input_size = len(all_letters)
    hidden_size = 128
    output_size = categorynum

    my_lstm = MyLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    my_crossentrophy = nn.NLLLoss()
    my_adam = optim.Adam(my_lstm.parameters(), lr=my_lr)

    # 定义训练日志的参数
    start_time = time.time()  # 开始时间
    total_iter_num = 0  # 已经训练的样本数量
    total_loss = 0.0  # 已经训练的样本总损失
    total_loss_list = []  # 每迭代100个样本, 计算平均损失并保存列表中
    total_acc_num = 0  # 已经训练的样本中预测正确的样本数量
    total_acc_list = []  # 每迭代100个样本, 计算平均准确率并保存列表中

    for epoch_idx in range(epochs):
        # 实例化Dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

        # 开始内部迭代数据
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # x -> [1, 6, 57]
            h0, c0 = my_lstm.inithidden()  # h0 = [1, 1, 128]
            output, hn, cn = my_lstm(x[0], h0, c0)  # output -> [1, 18]

            # 计算损失
            my_loss = my_crossentrophy(output, y)

            # 梯度清零
            my_adam.zero_grad()

            # 反向传播
            my_loss.backward()

            # 参数更新
            my_adam.step()
            
            total_iter_num += 1
            total_loss += my_loss.item()
            # 计算已经训练的样本预测正确的个数
            tag = (1 if torch.argmax(output).item() == y.item() else 0)
            total_acc_num += tag

            # 每隔100次计算一下平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                # 平均准确率
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 每隔两千步打印一下训练日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                use_time = time.time()
                print('轮次:%d, 损失:%.6f, 时间:%d, 准确率:%.4f' %(epoch_idx+1, temp_loss, use_time - start_time, temp_acc))
        
        # 每一轮都保存模型
        torch.save(my_lstm.state_dict(), f="./model/lstm_%d.bin" % (epoch_idx+1))
    
    # 计算总时间
    all_time = time.time() - start_time
    dict1 = {
        "total_loss_list": total_loss_list,
        "all_time": all_time,
        "total_acc_list": total_acc_list
    }
    with open("./data/lstm.json", mode="w", encoding='utf-8') as f:
        f.write(json.dumps(dict1))
    return total_loss_list, all_time, total_acc_list


def train_gru():
    my_list_x, my_list_y = read_data("./data/name_classfication.txt")
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    input_size = len(all_letters)
    hidden_size = 128
    output_size = categorynum

    my_gru = MyGRU(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    my_crossentrophy = nn.NLLLoss()
    my_adam = optim.Adam(my_gru.parameters(), lr=my_lr)

    # 定义训练日志的参数
    start_time = time.time()  # 开始时间
    total_iter_num = 0  # 已经训练的样本数量
    total_loss = 0.0  # 已经训练的样本总损失
    total_loss_list = []  # 每迭代100个样本, 计算平均损失并保存列表中
    total_acc_num = 0  # 已经训练的样本中预测正确的样本数量
    total_acc_list = []  # 每迭代100个样本, 计算平均准确率并保存列表中

    for epoch_idx in range(epochs):
        # 实例化Dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

        # 开始内部迭代数据
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # x -> [1, 6, 57]
            h0 = my_gru.inithidden()  # h0 = [1, 1, 128]
            output, hn = my_gru(x[0], h0)  # output -> [1, 18]

            # 计算损失
            my_loss = my_crossentrophy(output, y)

            # 梯度清零
            my_adam.zero_grad()

            # 反向传播
            my_loss.backward()

            # 参数更新
            my_adam.step()
            
            total_iter_num += 1
            total_loss += my_loss.item()
            # 计算已经训练的样本预测正确的个数
            tag = (1 if torch.argmax(output).item() == y.item() else 0)
            total_acc_num += tag

            # 每隔100次计算一下平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                # 平均准确率
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 每隔两千步打印一下训练日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                use_time = time.time()
                print('轮次:%d, 损失:%.6f, 时间:%d, 准确率:%.4f' %(epoch_idx+1, temp_loss, use_time - start_time, temp_acc))
        
        # 每一轮都保存模型
        torch.save(my_gru.state_dict(), f="./model/gru_%d.bin" % (epoch_idx+1))
    
    # 计算总时间
    all_time = time.time() - start_time
    dict1 = {
        "total_loss_list": total_loss_list,
        "all_time": all_time,
        "total_acc_list": total_acc_list
    }
    with open("./data/gru.json", mode="w", encoding='utf-8') as f:
        f.write(json.dumps(dict1))
    return total_loss_list, all_time, total_acc_list


def read_json(pathname):
    # 获取数据
    with open(pathname, mode="r", encoding="utf-8") as f:
        line = f.readline()
    dict1 = json.loads(line)
    total_loss_list = dict1["total_loss_list"]
    all_time = dict1["all_time"]
    total_acc_list = dict1["total_acc_list"]
    return total_loss_list, all_time, total_acc_list


# 画损失曲线图
def show_picture():
    # 获取数据
    rnn_total_loss_list, rnn_all_time, rnn_total_acc_list = read_json("./data/rnn.json")
    gru_total_loss_list, gru_all_time, gru_total_acc_list = read_json("./data/gru.json")
    lstm_total_loss_list, lstm_all_time, lstm_total_acc_list = read_json("./data/lstm.json")

    # 画损失对比曲线
    plt.figure(0)
    plt.plot(rnn_total_loss_list, label="RNN", color="red")
    plt.plot(gru_total_loss_list, label="GRU", color="green")
    plt.plot(lstm_total_loss_list, label="LSTM", color="blue")
    plt.legend(loc="best")
    plt.savefig("./image/rnn_loss.png")
    plt.show()

    # 画运行时间对比柱状图
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_all_time, lstm_all_time, gru_all_time]
    plt.bar(x_data, y_data, label=x_data, color=["r", "g", "b"])
    plt.legend(loc="best")
    plt.savefig("./image/rnn_time.png")
    plt.show()

    # 准确率曲线
    plt.figure(2)
    plt.plot(rnn_total_acc_list, label="RNN", color="red")
    plt.plot(gru_total_acc_list, label="GRU", color="green")
    plt.plot(lstm_total_acc_list, label="LSTM", color="blue")
    plt.legend(loc="best")
    plt.savefig("./image/rnn_acc.png")
    plt.show()


# 定义模型训练保存路径
my_rnn_path = "./model/rnn_1.bin"
my_lstm_path = "./model/lstm_1.bin"
my_gru_path = "./model/gru_1.bin"


# 定义文本转换为张量的函数
def line2tensor(x):
    tensor__x = torch.zeros(len(x), len(all_letters))
    for li, letter in enumerate(x):
        tensor__x[li][all_letters.find(letter)] = 1
    return tensor__x


# 定义rnn预测函数
def my_rnn_predict(x):
    # 将人名转化为张量函数
    tensor_x = line2tensor(x)

    # 实例化模型, 并加载训练好的参数
    my_rnn = MyRNN(input_size=len(all_letters), hidden_size=128, output_size=categorynum)
    my_rnn.load_state_dict(torch.load(my_rnn_path, weights_only=True))

    # 开始预测
    with torch.no_grad():
        h0 = my_rnn.inithidden()
        output, hn = my_rnn(tensor_x, h0)
        # 获取output结果中, 概率值最大的前三个值, 并且包括其索引
        topv, topi = torch.topk(input=output, k=3, dim=-1)

        print("rnn: ")
        for i in range(3):
            value = topv[0][i]
            idx = topi[0][i]
            category = categorys[idx]
            print(f"当前人名是: {x}, value: {value}, 国家类别: {category}")


def my_lstm_predict(x):
    # 将人名转化为张量函数
    tensor_x = line2tensor(x)

    # 实例化模型, 并加载训练好的参数
    my_lstm = MyLSTM(input_size=len(all_letters), hidden_size=128, output_size=categorynum)
    my_lstm.load_state_dict(torch.load(my_lstm_path, weights_only=True))

    # 开始预测
    with torch.no_grad():
        h0, c0 = my_lstm.inithidden()
        output, hn, cn = my_lstm(tensor_x, h0, c0)
        # 获取output结果中, 概率值最大的前三个值, 并且包括其索引
        topv, topi = torch.topk(input=output, k=3, dim=-1)

        print("lstm: ")
        for i in range(3):
            value = topv[0][i]
            idx = topi[0][i]
            category = categorys[idx]
            print(f"当前人名是: {x}, value: {value}, 国家类别: {category}")


def my_gru_predict(x):
    # 将人名转化为张量函数
    tensor_x = line2tensor(x)

    # 实例化模型, 并加载训练好的参数
    my_gru = MyGRU(input_size=len(all_letters), hidden_size=128, output_size=categorynum)
    my_gru.load_state_dict(torch.load(my_gru_path, weights_only=True))

    # 开始预测
    with torch.no_grad():
        h0 = my_gru.inithidden()
        output, hn = my_gru(tensor_x, h0)
        # 获取output结果中, 概率值最大的前三个值, 并且包括其索引
        topv, topi = torch.topk(input=output, k=3, dim=-1)

        print("gru: ")
        for i in range(3):
            value = topv[0][i]
            idx = topi[0][i]
            category = categorys[idx]
            print(f"当前人名是: {x}, value: {value}, 国家类别: {category}")



if __name__ == "__main__":
    # train_rnn()  # 0.6591
    # train_lstm()  # 0.6999
    # train_gru()  # 0.7199
    # show_picture()
    # my_rnn_predict("Wang")
    # my_lstm_predict("zhang")
    # my_gru_predict("Song")
    for func in [my_rnn_predict, my_lstm_predict, my_gru_predict]:
        func("Chuang")

    ...
