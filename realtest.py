from datasets import load_dataset
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer
from util.format_util import FormatConverter
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
import torch

#测试embeddingmodel
queries = [
    "What is the capital of China?",
    # "Explain gravity",
    # "What is the capital of China?"
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The capital of America is NewYork",
    "The capital of Japan is Tokyo"
]
# query_embeddings = model.encode(queries, prompt_name="query")
query_embeddings = model.encode(queries)
document_embeddings = model.encode(documents)
similarity = model.similarity(query_embeddings, document_embeddings)
top_values, top_indices = torch.topk(similarity, k=3)
top_chunks = [(idx, score) for idx, score in zip(top_indices.tolist(),top_values.tolist())]
for i in top_indices.tolist():
    print(i)
print(similarity)
print(top_values.tolist())
print(top_indices.tolist())
print(top_chunks)
# result=FormatConverter.remove_brackets_and_content("hello <end> nihao")
# print(res)
