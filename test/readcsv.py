def load_antonym_lexicon(path):
    antonym_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            cols = line.split(",")

            if len(cols) < 3:
                continue

            word = cols[0].strip().lower()

            # 第二列是同义词，不用
            antonyms_raw = cols[2].strip()

            # 反义词用 # 分隔
            antonyms = [a.strip().lower() for a in antonyms_raw.split("#") if a.strip()]

            # 保存到 dict
            if antonyms:
                antonym_dict[word] = antonyms

    return antonym_dict
dict_final=load_antonym_lexicon("syn-ant.csv")

print(load_antonym_lexicon("syn-ant.csv"))
# print(dict_final["slow"])
print(dict_final["decline"])

# NEGATION_PATTERNS = [
#     "not",
#     "n't",
#     "no",
#     "never",
#     "none",
#     "cannot",
#     "can't",
#     "does not",
#     "do not",
#     "did not",
#     "is not",
#     "was not",
#     "are not",
#     "were not",
#     "has not",
#     "have not",
#     "had not",
#     "cannot",
#     "without",
# ]
#
# def has_negation(text: str) -> bool:
#     text = text.lower().strip()
#     for pat in NEGATION_PATTERNS:
#         if pat in text:
#             return True
#     return False
#
# print(has_negation("is rotating faster"))        # False
# print(has_negation("is not rotating faster"))    # True
# print(has_negation("does not cause"))            # True
# print(has_negation("never happens"))             # True
# print(has_negation("isn't rotating faster"))        # False