import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import Dataset, DataLoader
import string
import matplotlib.pyplot as plt


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
            

if __name__ == "__main__":
    my_list_x, my_list_y = read_data("./data/name_classfication.txt")
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    for x, y in my_dataloader:
        print(x.shape)
        print(y.shape)
        break




        





