import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification


def test_classfication():
    # 1. 加载tokenizer分词器
    cls_tokenizer = AutoTokenizer.from_pretrained("./model/bert-base-chinese")

    # 2. 加载模型
    cls_model = AutoModelForSequenceClassification.from_pretrained("./model/bert-base-chinese")
    # print(cls_model)

    # 3. 准备数据
    message = "我爱北京天安门"
    # 3.1 对上述字符串进行编码, 变成向量送给模型
    # padding 补齐到最大长度, truncation 截断
    inputs = cls_tokenizer.encode(message, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
    # inputs = cls_tokenizer.encode(message, padding="max_length", truncation=True, max_length=20)

    # 4. 设置模型为评估模式
    cls_model.eval()

    # 5. 将数据送入model
    result = cls_model(inputs)

    # 打印文本分类的结果
    print(result)
    logits = result.logits
    topv, topi = torch.topk(logits, k=1)
    print(topv)
    print(topi)


def test_features_extraction():
    fe_tokenizer = AutoTokenizer.from_pretrained("./model/bert-base-chinese")
    fe_model = AutoModel.from_pretrained("./model/bert-base-chinese")

    message = ['你是谁', '人生该如何起头']
    inputs = fe_tokenizer.encode_plus(message, padding="max_length", max_length=20, truncation=True, return_tensors="pt")
    print(f"inputs: {inputs}")

    fe_model.eval()

    result = fe_model(**inputs)
    print(result)
    print(f"last_hidden_state: {result['last_hidden_state'].shape}")
    print(f"pooler_output: {result['pooler_output'].shape}")


def test_fill_mask():
    fm_tokenizer = AutoTokenizer.from_pretrained("./model/chinese-bert-wwm")
    fm_model = AutoModelForMaskedLM.from_pretrained("./model/chinese-bert-wwm")

    message = "我想明天去[MASK]家吃饭."
    inputs = fm_tokenizer.encode_plus(message, return_tensors="pt")
    print(f"inputs: {inputs}")

    fm_model.eval()
    results = fm_model(**inputs)
    logits = results["logits"]
    print(f"results: {results}")
    print(f"logits: {logits.shape}")  # [1, 12, 21128]

    # 取出mask位置, 对应预测最大概率的索引值
    mask_pred_idx = torch.argmax(input=logits[0, 6]).item()
    print(f"最终预测结果: {fm_tokenizer.convert_ids_to_tokens([mask_pred_idx])}")


def test_question_answering():
    qa_tokenizer = AutoTokenizer.from_pretrained("./model/chinese_pretrain_mrc_roberta_wwm_ext_large")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("./model/chinese_pretrain_mrc_roberta_wwm_ext_large")

    context = '我叫张三 我是一个程序员 我的喜好是打篮球'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']

    # 一个问题一个问题去解决
    for question in questions:
        input = qa_tokenizer.encode_plus(question, context, return_tensors="pt")
        print(f"input: {input}")
        qa_model.eval()
        output = qa_model(**input)
        print(f"output: {output}")
        print(f"start_logits: {output['start_logits'].shape}")  # [1, 26]
        print(f"end_logits: {output['end_logits'].shape}")  # [1, 26]
        # 找出start_logits预测最大概率值对应的索引
        start_idx = torch.argmax(output["start_logits"], dim=-1).item()
        end_idx = torch.argmax(output["end_logits"], dim=-1).item() + 1

        # 转化为真实结果
        result = qa_tokenizer.convert_ids_to_tokens(input["input_ids"][0][start_idx: end_idx])
        print(f"result: {''.join(result)}")
        

def test_summary():
    s_tokenizer = AutoTokenizer.from_pretrained("./model/distilbart-cnn-12-6")
    s_model = AutoModelForSeq2SeqLM.from_pretrained("./model/distilbart-cnn-12-6")

    text = " In Executive Order 14257 of April 2, 2025 (Regulating Imports with a Reciprocal Tariff to Rectify Trade Practices that Contribute to Large and Persistent Annual United States Goods Trade Deficits), I declared a national emergency arising from conditions reflected in large and persistent annual U.S. goods trade deficits, and imposed additional ad valorem duties that I deemed necessary and appropriate to deal with that unusual and extraordinary threat, which has its source in whole or substantial part outside the United States, to the national security and economy of the United States.  Section 4(b) of Executive Order 14257 provided that “[s]hould any trading partner retaliate against the United States in response to this action through import duties on U.S. exports or other measures, I may further modify the [Harmonized Tariff Schedule of the United States] to increase or expand in scope the duties imposed under this order to ensure the efficacy of this action.”  I further declared pursuant to Executive Order 14256 of April 2, 2025 (Further Amendment to Duties Addressing the Synthetic Opioid Supply Chain in the People’s Republic of China as Applied to Low-Value Imports) that duty-free de minimis treatment on articles described in section 2(a) of Executive Order 14195 is no longer available effective at 12:01 a.m. eastern daylight time on May 2, 2025."

    inputs = s_tokenizer.encode_plus(text, return_tensors='pt')
    print(f"inputs: {inputs}")

    s_model.eval()
    outputs = s_model.generate(**inputs)
    print(f"outputs: {outputs}")
    print(f"outputs_shape: {outputs.shape}")

    # 解码
    # skip_special_tokens 跳过特殊字符
    # clean_up_tokenization_spaces 将标点符号和单词分隔开
    result = s_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(result)


# Named Entity Recognition
def test_NER(): 
    ner_tokenizer = AutoTokenizer.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
    ner_model = AutoModelForTokenClassification.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
    ner_config = AutoConfig.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")  # 加载配置文件

    message = "我爱北京天安门，天安门上太阳升"

    # 数据编码
    inputs = ner_tokenizer.encode_plus(message, return_tensors='pt')

    # 将数据送入模型
    ner_model.eval()
    result = ner_model(**inputs)
    logits = result["logits"]
    # print(f"result: {result}")
    print(f"logits: {result['logits'].shape}")  # [1, 17, 32]  32个类别
    
    # 得到原始的句子
    input_tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 对每个单词找出对应的预测标签
    for token, value in zip(input_tokens, logits[0]):
        # print(f"token: {token}")
        # print(f"valuel: {value}")
        if token in ner_tokenizer.all_special_tokens:
            continue
        idx = torch.argmax(value).item()
        print(f"{token}: {ner_config.id2label[idx]}")


if __name__ == "__main__":
    # test_classfication()
    # test_features_extraction()
    # test_fill_mask()
    # test_question_answering()
    # test_summary()
    test_NER()
    ...


