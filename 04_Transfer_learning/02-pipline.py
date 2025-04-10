from transformers import pipeline
import numpy  as np


# 实现文本分类
def test_classification():
    # 直接调用pipline方法获得模型
    model = pipeline(task="text-classification", model="./model/bert-base-chinese")
    # 根据model去预测
    result = model("我爱北京天安门，天安门上太阳升。")
    print(f"文本分类的结果: {result}")


# 特征抽取任务
def test_features_extraction():
    model = pipeline(task="feature-extraction", model="./model/bert-base-chinese")
    result = model("人生该如何起头")
    print(f"文本分类的结果: {np.array(result).shape}")
    # # 7个字变成9个字原因: [CLS] 人 生 该 如 何 起 头 [SEP]


# 实现完形填空任务
def test_fill_mask():
    model = pipeline(task="fill-mask", model="./model/chinese-bert-wwm")
    input = "我想去[MASK]家吃饭"
    result = model(inputs=input)
    print(result)


# 阅读理解任务
def test_question_answering():
    context = "我叫张三，我是一个程序员，我的喜好是打篮球。"
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    model = pipeline(task="question-answering", model="./model/chinese_pretrain_mrc_roberta_wwm_ext_large")
    result = model(context=context, question=questions)
    print(result)


# 文本摘要任务
def test_summary():
    model = pipeline(task="summarization", model="./model/distilbart-cnn-12-6")
    context = " In Executive Order 14257 of April 2, 2025 (Regulating Imports with a Reciprocal Tariff to Rectify Trade Practices that Contribute to Large and Persistent Annual United States Goods Trade Deficits), I declared a national emergency arising from conditions reflected in large and persistent annual U.S. goods trade deficits, and imposed additional ad valorem duties that I deemed necessary and appropriate to deal with that unusual and extraordinary threat, which has its source in whole or substantial part outside the United States, to the national security and economy of the United States.  Section 4(b) of Executive Order 14257 provided that “[s]hould any trading partner retaliate against the United States in response to this action through import duties on U.S. exports or other measures, I may further modify the [Harmonized Tariff Schedule of the United States] to increase or expand in scope the duties imposed under this order to ensure the efficacy of this action.”  I further declared pursuant to Executive Order 14256 of April 2, 2025 (Further Amendment to Duties Addressing the Synthetic Opioid Supply Chain in the People’s Republic of China as Applied to Low-Value Imports) that duty-free de minimis treatment on articles described in section 2(a) of Executive Order 14195 is no longer available effective at 12:01 a.m. eastern daylight time on May 2, 2025."
    result = model(context)
    print(result)
    

#  NER任务
def test_NER():
    model = pipeline(task="token-classification", model="./model/roberta-base-finetuned-cluener2020-chinese")
    result = model("我在广州")
    print(result)

    
if __name__ == "__main__":
    # test_classification()
    # test_features_extraction()
    # test_fill_mask()
    # test_question_answering()
    # test_summary()
    test_NER()
    ...