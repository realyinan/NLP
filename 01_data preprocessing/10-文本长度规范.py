from keras.preprocessing import sequence


# 句子长度分布90%样本符合的长度
# padding: 填充 pre, post
# truncating: 截断 pre, post
cutlen = 10
def padding(x_train):
    return sequence.pad_sequences(x_train, cutlen, padding="post", truncating="post")


def my_padding(x_train):
    new_list = []
    max_len = 10
    for x in x_train:
        if len(x) >= max_len:
            new_list.append(x[:10])
        else:
            counts = max_len - len(x)
            list1 = x + [0]*counts
            new_list.append(list1)
    return new_list


if __name__ == "__main__":
    x_train = [
        [1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1], 
        [2, 32, 1, 23, 1]]
    print(padding(x_train))
    print(my_padding(x_train))