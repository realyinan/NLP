import fasttext


def fasttest_01():
    # 直接开始训练
    model = fasttext.train_unsupervised("./data/fil9")

    # 保存模型
    model.save_model("./data/fil9.bin")


def fasttext_02():
    # 加载模型
    model = fasttext.load_model("./data/fil9.bin")

    # 直接获取某个词的向量
    result = model.get_word_vector("tiger")
    print(type(result))
    print(result)
    print(result.shape)

    # 检验模型效果
    result = model.get_nearest_neighbors("tennis")
    print(result)

def fasttext_03():
    # 修改参数
    model = fasttext.train_unsupervised("./data/fil9_aaa", "cbow", dim=300, lr=0.1, epoch=1)
    model.save_model("./data/fil_aaa.bin")




if __name__ == "__main__":
    # fasttest_01()
    fasttext_02()
    # fasttext_03()