from sentence_transformers import SentenceTransformer
import json

from sympy import false
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk

import spacy
nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")


# print(A["head"])

def embed_texts(texts, model):
    """批量 embedding，返回 numpy 数组 (N, D)"""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(emb_a, emb_b,model) -> float:
    return model.similarity(emb_a,emb_b)

def triplet_similarity(
    tA,
    tB,
    model,
    w_head: float = 0.4,
    w_rel: float = 0.2,
    w_tail: float = 0.4,
) -> float:
    """
    对两个三元组进行相似度计算：
        head vs head
        relation vs relation
        tail vs tail
    最终用权重加权融合。
    """

    # 分别嵌入三个部分
    embs = embed_texts(
        [tA["head"], tA["relation"], tA["tail"],
         tB["head"], tB["relation"], tB["tail"]],
        model
    )
    hA, rA, tA_emb = embs[0], embs[1], embs[2]
    hB, rB, tB_emb = embs[3], embs[4], embs[5]

    # 计算三部分相似度
    sim_head = cosine_sim(hA, hB,model)
    sim_rel  = cosine_sim(rA, rB,model)
    sim_tail = cosine_sim(tA_emb, tB_emb,model)

    # 加权融合
    final_sim = w_head * sim_head + w_rel * sim_rel + w_tail * sim_tail
    return float(final_sim)

def most_similar_triple(tripleA,tripleArr,model):
    max_index=0
    max_similarity=0
    for index,triple in enumerate(tripleArr):
        similarity=triplet_similarity(tripleA,triple,model)
        # print("index",index)
        # print("similarity",similarity)
        if similarity>max_similarity:
            max_similarity=similarity
            max_index=index
            # print("max_index",max_index)
    return max_index

NEGATION_PATTERNS = [
    "not",
    "n't",
    "no",
    "never",
    "none",
    "cannot",
    "can't",
    "does not",
    "do not",
    "did not",
    "is not",
    "was not",
    "are not",
    "were not",
    "has not",
    "have not",
    "had not",
    "cannot",
    "without",
]
#判断text里是否含有否定词
def has_negation(text: str) -> bool:
    text = text.lower().strip()
    for pat in NEGATION_PATTERNS:
        if pat in text:
            return True
    return False

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

#将text转变为词元的数组
def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

#判断两个三元组里是否存在冲突
def single_conflict_detection(tA,tB,model,dict_final):
    relation_conflict_flag = False
    negation_conflict_flag=False
    embs = embed_texts(
        [tA["head"], tA["relation"], tA["tail"],
         tB["head"], tB["relation"], tB["tail"]],
        model
    )
    hA, rA, tA_emb = embs[0], embs[1], embs[2]
    hB, rB, tB_emb = embs[3], embs[4], embs[5]

    #calculate similarity
    sim_head = cosine_sim(hA, hB,model)
    sim_rel  = cosine_sim(rA, rB,model)
    sim_tail = cosine_sim(tA_emb, tB_emb,model)


    # relation上是否有冲突

    flag_a = has_negation(tA["relation"])
    flag_b = has_negation(tB["relation"])
    if flag_a != flag_b:
        negation_conflict_flag=True

    r1_words = lemmatize(tA["relation"])
    r2_words = lemmatize(tB["relation"])
    for w1 in r1_words:
        for w2 in r2_words:
            if w1 in dict_final and w2 in dict_final[w1]:
                relation_conflict_flag=True
            if w2 in dict_final and w1 in dict_final[w2]:
                relation_conflict_flag = True

    # head,tail高度相似的情况
    if sim_head>0.75 and sim_tail>0.75:
        if relation_conflict_flag!=negation_conflict_flag:
            return True
    # head和relation高度
    elif sim_head>0.75 and sim_rel>0.8 and (relation_conflict_flag==negation_conflict_flag):
        if sim_tail<0.45:
            return True
    elif sim_tail>0.75 and sim_rel>0.8 and (relation_conflict_flag==negation_conflict_flag):
        if sim_head<0.45:
            return True

    return false

#获取两个多三元组的冲突检测得分
def group_conflict_detection_score(triple_group_A,triple_group_B,model,dict_final):
    score=0
    for triple in triple_group_A:
        index=most_similar_triple(triple,triple_group_B,model)
        print("triple",triple)
        print("index,",index)
        print(single_conflict_detection(triple,triple_group_B[index],model, dict_final))
        if single_conflict_detection(triple,triple_group_B[index],model, dict_final):
            score+=1

    return score
A = [
    {"head": "the planet", "relation": "is rotating slower", "tail": "after the impact"},
    {"head": "rotation speed", "relation": "increases", "tail": "gravitational pull"},
]

B = [
    {"head": "rotation speed", "relation": "decreases", "tail": "gravitational pull"},
    {"head": "the sun", "relation": "is rotating faster", "tail": "after the impact"},
    {"head": "the child", "relation": "love", "tail": "parents"},
]
score=group_conflict_detection_score(A,B,embedding_model,dict_final)
print("score ",score)
# group_conflict_detection_score(A,B,embedding_model,dict_final)
# A = {"head": "the planet", "relation": "is not rotating faster", "tail": "after the impact"}
# B = {"head": "the sun", "relation": "is rotating faster", "tail": "after the impact"}
# C = {"head": "rotation speed", "relation": "increases", "tail": "gravitational pull"}
# A = {"head": "the planet", "relation": "is not rotating faster", "tail": "after the impact"}
# # B = {"head": "the sun", "relation": "is rotating faster", "tail": "after the impact"}
# C = {"head": "rotation speed", "relation": "increases", "tail": "gravitational pull"}

# tA= {"head": "rotation speed", "relation": "increases", "tail": "gravitational pull"}
# tB= {"head": "rotation speed", "relation": "decreases", "tail": "gravitational pull"}
# flag=single_conflict_detection(tA,tB,embedding_model,dict_final)
# print(flag)




# text_a="A is B"
# text_b="A is not B"
# emb_a=embed_texts(text_a,embedding_model)
# emb_b=embed_texts(text_b,embedding_model)
# print(cosine_sim(emb_a,emb_b,embedding_model))
# print(triplet_similarity(A,B,model=embedding_model))
# print(triplet_similarity(A,C,model=embedding_model))
# print(most_similar_triple(A,B,model=embedding_model))

