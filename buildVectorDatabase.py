import pickle

import faiss
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import numpy as np
dataset = load_from_disk("chunk_dataset")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print(dataset)
texts, ids = [], []
for item in dataset:
    for chunk in item["chunks"]:
        texts.append(chunk)
        ids.append(item["id"])   # ✅ 保留来源 id

print(f"一共 {len(texts)} 个文本块")
print(ids)
embeddings = model.encode(
    texts,
    batch_size=32,             # ✅ 防止显存溢出
    show_progress_bar=True,    # ✅ 显示进度
    convert_to_numpy=True,     # ✅ 转成 numpy，方便存 FAISS
    normalize_embeddings=True  # ✅ 若后续用内积搜索，相当于余弦相似度
)
print(f"向量 shape: {embeddings.shape}")

# index = faiss.IndexFlatL2(dimension)
embeddings = np.array(embeddings).astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"✅ 向量库中共有 {index.ntotal} 条向量")

# 6️⃣ 保存索引与元数据
faiss.write_index(index, "chunk_index.faiss")
with open("chunk_metadata.pkl", "wb") as f:
    pickle.dump({"texts": texts, "ids": ids}, f)

print("💾 已保存 FAISS 索引与元数据！")