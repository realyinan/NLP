import torch
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn
import tensorflow as tf
import tensorboard as tb


def nnembedding_show():
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    # 对句子分词
    word_list = []
    for s in sentences:
        word_list.append(jieba.lcut(s))
    # print(word_list)

    # 对每个词进行词表映射
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(word_list)
    # print(my_tokenizer.word_index)
    # print(my_tokenizer.index_word)

    # 获取所有词语
    my_token_list = my_tokenizer.index_word.values()
    # print(my_token_list)

    # 打印文本数值化以后的句子
    sentence2id = my_tokenizer.texts_to_sequences(word_list)
    # print(sentence2id)

    # 创建embedding层
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    # print(embed.weight)
    # print(embed.weight.shape)

    # 可视化展示
    # my_summary = SummaryWriter()
    # my_summary.add_embedding(embed.weight, my_token_list)
    # my_summary.close()

    # 取出每个单词对应的向量展示
    for idx in range(len(my_tokenizer.index_word)):
        temp_vector = embed(torch.tensor(idx))
        word = my_tokenizer.index_word[idx+1]
        print(f"{word}: {temp_vector.detach().numpy()}")



if __name__ == "__main__":
    nnembedding_show()