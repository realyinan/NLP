from itertools import chain
import jieba
import pandas as pd


def get_vocabs():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    train_vocabs = set(chain(*map(lambda x: jieba.lcut(x), train_data["text"])))  # "*" 解包：把一个“列表”变成多个“参数”传进去
    print(len(train_vocabs))

    text_vocabs = set(chain(*map(lambda x: jieba.lcut(x), test_data["text"])))
    print(len(text_vocabs))









if __name__ == "__main__":
    get_vocabs()