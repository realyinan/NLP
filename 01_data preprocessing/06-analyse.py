import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# 标签分布
def label_counts():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    # 画图
    sns.countplot(x="label", data=train_data)
    plt.title("train_data")
    plt.show()

    sns.countplot(x="label", data=test_data)
    plt.title("test_data")
    plt.show()


# 句子长度分布
def length_counts():
    train_data = pd.read_csv("./data/train.csv")
    train_data["sequence_length"] = list(map(lambda x: len(x), train_data['text']))

    test_data = pd.read_csv("./data/test.csv")
    test_data["sequence_length"] = list(map(lambda x: len(x), test_data['text']))

    # 训练集句子长度画图
    sns.countplot(x="sequence_length", data=train_data, hue="sequence_length")
    plt.xticks([])
    plt.show()

    # 训练集句子长度画图--曲线图
    sns.displot(x="sequence_length", data=train_data, kind="kde")
    plt.xticks([])
    plt.show()  

    # 验证集句子长度画图
    sns.countplot(x="sequence_length", data=test_data, hue="sequence_length")
    plt.xticks([])
    plt.show()

    # 验证集句子长度画图--曲线图
    sns.displot(x="sequence_length", data=test_data, kind="kde")
    plt.xticks([])
    plt.show()


# 正负样本句子长度散点图
def stripplot():
    train_data = pd.read_csv("./data/train.csv")
    train_data["sequence_length"] = list(map(lambda x: len(x), train_data['text']))

    test_data = pd.read_csv("./data/test.csv")
    test_data["sequence_length"] = list(map(lambda x: len(x), test_data['text']))

    # 画散点图
    sns.stripplot(y="sequence_length", x="label", data=train_data, hue="label")
    plt.title("train_data")
    plt.show()

    sns.stripplot(y="sequence_length", x="label", data=test_data, hue="label")
    plt.title("test_data")
    plt.show()



if __name__ == "__main__":
    # label_counts()
    # length_counts()
    stripplot()