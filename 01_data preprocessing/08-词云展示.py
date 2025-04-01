import pandas as pd
import jieba
import jieba.posseg as pseg
from wordcloud import WordCloud
from itertools import chain
from matplotlib import pyplot as plt


def get_a_list(x):
    r = []
    for g in pseg.lcut(x):
        if g.flag == 'a':
            r.append(g.word)
    return r


def get_word_cloud(kwyword_list):
    # 实例化一个词云对象
    word_cloud = WordCloud(font_path="./data/SimHei.ttf", max_words=100, background_color="white")

    # 准备数据
    keyword_str = " ".join(kwyword_list)
    word_cloud.generate(keyword_str)

    # 画图
    plt.figure()
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()



# 高频词云展示
def word_cloud():
    # 读取数据
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    # 获取训练集的正样本
    p_train_data = train_data[train_data["label"] == 1]["text"]

    # 获取正样本所有的形容词
    p__train_a_vocab = list(chain(*map(lambda x: get_a_list(x), p_train_data)))
    # print(len(p__train_a_vocab))

    # 词云展示
    get_word_cloud(p__train_a_vocab)



if __name__ == "__main__":
    word_cloud()