from transformers import BertTokenizer, BertForMaskedLM
import torch


def test_fill_mask():
    fm_tokenizer = BertTokenizer.from_pretrained("./model/chinese-bert-wwm")
    fm_model = BertForMaskedLM.from_pretrained("./model/chinese-bert-wwm")

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


if __name__ == "__main__":
    test_fill_mask()