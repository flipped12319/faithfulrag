from datasets import load_dataset
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# dataset = load_dataset("wikipedia", "20220301.en", split="train")
dataset = load_dataset("eric-xiang/FaithfulRAG-Dataset")


# 用和 Qwen/Qwen3-Embedding-0.6B 对应的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
print(dataset)

#
def chunk_text(text, chunk_size=128, overlap=32):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokenizer.decode(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def split_into_chunks(batch):
    all_chunks = []
    for text in batch["context"]:  # batch["text"] 是 list[str]
        all_chunks.append(chunk_text(text, chunk_size=20, overlap=5))
    return {"chunks": all_chunks}
#
chunked_dataset = dataset["train"].map(split_into_chunks, batched=True,batch_size=100, num_proc=8, remove_columns=["question", "answer","answerKey","choices","justification","num of options",'context'])
# print(chunked_dataset['chunks'][0])
# print(chunked_dataset["train"][0])
# print(chunked_dataset)
chunked_dataset = chunked_dataset.flatten()
print(chunked_dataset[0])
chunked_dataset.save_to_disk("chunk_dataset")












# text = "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
#
# chunks = chunk_by_tokens(text, chunk_size=20, overlap=5)
# print(chunks)

# queries = [
#     "What is the capital of China?",
#     "Explain gravity",
# ]
# documents = [
#     "The capital of China is Beijing.",
#     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
# ]
# query_embeddings = model.encode(queries, prompt_name="query")
# document_embeddings = model.encode(documents)
# similarity = model.similarity(query_embeddings, document_embeddings)
# print(similarity)

# embeddings = model.encode(chunks)
# print(embeddings.shape)
# print(embeddings)
# embeddings = np.array(embeddings).astype("float32")


# 建立索引
# d = embeddings.shape[1]  # 向量维度
# index = faiss.IndexFlatL2(d)
# index.add(embeddings)  # 加入向量
# print("数据库大小:", index.ntotal)

# batch_size = 1000
# for i in range(0, len(dataset), batch_size):
#     batch_texts = dataset[i: i+batch_size]["text"]
#     batch_embeddings = model.encode(batch_texts, batch_size=64)
#     index.add(np.array(batch_embeddings).astype("float32"))

# 查询
# query = model.encode(["It gives weight to physical objects and is responsible "])
# D, I = index.search(np.array(query).astype("float32"), k=3)
# print("Top 3:", [chunks[i] for i in I[0]])
# print(I)
