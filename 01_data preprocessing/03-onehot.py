# 导入keras中的词汇映射器Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入用于对象保存与加载的joblib
import joblib

# onehot编码的生成
def onehot_gen():
    vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]
    # 实例化分词器
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(vocabs)

    for vocab in vocabs:
        zero_list = [0]*len(vocabs)
        idx = my_tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(f"当前单词{vocab}的one-hot编码是: {zero_list}")

    joblib.dump(my_tokenizer, "./data/my_tokenizer")

    print(my_tokenizer.word_index)
    print(my_tokenizer.index_word)


# onehot编码的使用
def onehot_use():
    vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]
    # 加载训练好的分词器
    my_tokenizer = joblib.load("./data/my_tokenizer")
    token = "王力宏"
    zero_list = [0]*len(vocabs)
    idx = my_tokenizer.word_index[token] - 1
    zero_list[idx] = 1
    print(f"当前单词的onehot编码: {zero_list}")




if __name__ == "__main__":
    # onehot_gen()
    onehot_use()