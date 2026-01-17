from datasets import load_dataset
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from qwen_instruct import LLMWithThinking
from module import GraphRAGModule
import json

from prompt import PromptGenerator

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
llm=LLMWithThinking(model,tokenizer)

# 针对hotpotqa数据集进行优化
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
dataArr=[]
for i in range(0,1):
    dataArr.append(ds["train"][i])

for i, data in enumerate(dataArr):
    text = ""
    # print(i)
    for sentences in dataArr[i]["context"]["sentences"]:
        for sentence in sentences:
            text += sentence + " "   # 正确写法
    text = text.strip()  # 去掉末尾空格

    dataArr[i]["context"] = text

print(dataArr[0]["context"])


graphRAGModule = GraphRAGModule(model, tokenizer)
# print(ds["train"][0]["context"]["sentences"])
chunk_inf=graphRAGModule.chunk_text(dataArr[0]["context"])
print(chunk_inf)

# 判断哪些chunk第一句话含有介词
PRONOUNS = ["it", "they", "he", "she", "this", "these", "those"]

def get_first_sentence(text: str) -> str:
    match = re.search(r'[.!?]', text)
    if match:
        return text[:match.end()]
    return text

def contains_pronoun(sentence: str) -> bool:
    sentence = sentence.lower()
    for p in PRONOUNS:
        if re.search(rf'\b{p}\b', sentence):
            return True
    return False

def find_chunks_with_pronouns(chunks):
    results = []
    chunk_index=[]
    chunk_arr=[]
    for idx, chunk in enumerate(chunks):
        first_sentence = get_first_sentence(chunk)
        if contains_pronoun(first_sentence):
            chunk_index.append(idx)
            chunk_arr.append(first_sentence)
    # results.append((chunk_index, chunk_arr))
    results.append(chunk_index)
    results.append(chunk_arr)
    return results

chunk_results = find_chunks_with_pronouns(chunk_inf[1])
print(chunk_results)
prompt_generator=PromptGenerator()
prompt_arr=[]
for value in chunk_results[0]:
    if value != 0:
        prompt=prompt_generator.coreference_resolution(chunk_pre=chunk_inf[1][value-1],chunk_target=chunk_inf[1][value])
        prompt_arr.append(prompt)
result = llm.generate_all(
    system_prompt="You are performing coreference resolution for knowledge graph construction.",
    prompt_list=prompt_arr)
print(result)