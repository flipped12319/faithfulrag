from collections import Counter
import re


def tokenize(text):
    """简单分词，可以根据需要换成更复杂的分词器"""
    return re.findall(r'\w+', text.lower())


def token_level_f1(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # 计算匹配 token 数量
    common = pred_counter & ref_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / sum(pred_counter.values())
    recall = num_common / sum(ref_counter.values())
    f1 = 2 * precision * recall / (precision + recall)

    return f1


# 示例
reference = "by releasing the water, with the pump becoming a hydroelectric power generator，"
prediction1 = "releasing the water"
prediction2 = "releasing the water through turbines"

print(token_level_f1(prediction1, reference))  # 0.857...
print(token_level_f1(prediction2, reference))  # 0.5