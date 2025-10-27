
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
with open("chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]   # FAISS 中每条向量对应的文本
ids = metadata["ids"]       # 对应的文档 id（可选）
# 从磁盘加载已保存的 FAISS 索引
index = faiss.read_index("chunk_index.faiss")

print(f"✅ 向量库中共有 {index.ntotal} 条向量")
print(ids)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

query = ["How are sedimentary rocks made?"]
query_embeddings = model.encode(query)
query_vec = np.array(query_embeddings).astype("float32")
k = 5  # 返回 top 5
D, I = index.search(query_vec, k)

print("相似度分数:", D)
print("匹配向量编号:", I)

# 打印对应的文本
for idx in I[0]:
    # print(idx,texts[idx])
    print(f'{ids[idx]}; {texts[idx]}\n')
