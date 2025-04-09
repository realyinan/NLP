import fasttext

# 原始语料进行训练
# model = fasttext.train_supervised("./data/cooking_train.txt")
# result= model.predict("Why not put knives in the dishwasher?")
# print(result)

# result = model.test("./data/cooking_valid.txt")
# print(result)  # (3000, 0.132, 0.05708519532939311)  (_, 精确率, 召回率)


# 清洗完数据
# model = fasttext.train_supervised("./data/cooking.pre.train")
# result = model.test("./data/cooking.pre.valid")
# print(f"模型效果: {result}")


# 清洗数据+增加训练轮数
# model = fasttext.train_supervised("./data/cooking.pre.train", epoch=25)
# result = model.test("./data/cooking.pre.valid")
# print(f"模型效果: {result}")


# 清洗数据+增加训练轮数+修改学习率
# model = fasttext.train_supervised("./data/cooking.pre.train", epoch=25, lr=1.0)
# result = model.test("./data/cooking.pre.valid")
# print(f"模型效果: {result}")


# 清洗数据+增加训练轮数+修改学习率+ 添加N-gram特征
# model = fasttext.train_supervised("./data/cooking.pre.train", epoch=25, lr=1.0, wordNgrams=2)
# result = model.test("./data/cooking.pre.valid")
# print(f"模型效果: {result}")


# 清洗数据+增加训练轮数+修改学习率+添加N-gram特征+修改损失计算方式  层次softmax
# model = fasttext.train_supervised("./data/cooking.pre.train", epoch=25, lr=1.0, wordNgrams=2, loss="hs") # hierarchical softmax
# result = model.test("./data/cooking.pre.valid")
# print(f"模型效果: {result}")


# 自动超参数调优
# model = fasttext.train_supervised("./data/cooking.pre.train", autotuneValidationFile="./data/cooking.pre.valid", autotuneDuration=60)
# result = model.test("./data/cooking.pre.valid")
# print(f"模型效果: {result}")


# 设置loss为多个二分类器输出
# model = fasttext.train_supervised("./data/cooking.pre.train", epoch=25, lr=0.2, wordNgrams=2, loss="ova")
# model.save_model("./model/cooking.bin")

# 参数k代表指定模型输出多少个标签, 默认为1, 这里设置为-1, 意味着尽可能多的输出.
# 参数threshold代表显示的标签概率阈值, 设置为0.5, 意味着显示概率大于0.5的标签
model = fasttext.load_model("./model/cooking.bin")
result = model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
print(result)
 