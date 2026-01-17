from datasets import load_dataset
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from qwen_instruct import LLMWithThinking
import json
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
llm=LLMWithThinking(model,tokenizer)
def chunk_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk_inf = []
    chunk_num = 0
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += (" " if current else "") + sent
        else:
            chunks.append(current)
            current = sent
            chunk_num = chunk_num + 1

    if current:
        chunks.append(current)
        chunk_num = chunk_num + 1

    chunk_inf.append(chunk_num)
    chunk_inf.append(chunks)

    return chunk_inf
# Login using e.g. `huggingface-cli login` to access this dataset
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
# print(ds["train"][0]["context"]["sentences"])
chunk_inf=chunk_text(dataArr[0]["context"])
print(chunk_inf)
for value in chunk_inf[1]:
    print(value)
    # 3. The replacement entity MUST appear explicitly in either chunk_pre or chunk_target
template="""
        You are performing a STRICT coreference resolution task that must output strictly valid JSON.

        ### Input Chunks
        - chunk_pre (context before target):
        {chunk_pre}

        - chunk_target (text to be modified):
        {chunk_target}

        ### Task
        Resolve coreferences in chunk_target by replacing pronouns with their explicit antecedents.

        You may modify ONLY pronoun tokens in chunk_target.
        All other changes are forbidden.

        Use chunk_pre and chunk_target ONLY as evidence.

        ### Procedure (MUST FOLLOW)

        1. Identify ALL pronouns in chunk_target
           (e.g., it, they, he, she, this, these, those).

        2. For each pronoun, find an antecedent that:
           - appears verbatim as a noun phrase
           - is unique and unambiguous,
           within chunk_pre or chunk_target.

        3. Replacement rule:
           - If a valid antecedent exists, Only replace the pronoun with the EXACT antecedent text.
           - If not, leave the pronoun unchanged.

        4. Output constraint (CRITICAL):
           - The output must be IDENTICAL to chunk_target,except for the replaced pronoun tokens.
           - Do NOT add, delete, reorder, or rewrite anything.

        5.Output a JSON object with EXACTLY one field:
        {{
          "resolved_chunk": "<the final modified chunk_target>"
        }}

        ### Output Requirements
        - Do NOT prepend, append, or merge chunk_pre with chunk_target.
        - Output ONLY the modified version of chunk_target.
        - Do NOT include explanations or comments.
        - If no valid antecedent exists, output chunk_target unchanged.
        - Removing a pronoun or changing sentence structure is STRICTLY FORBIDDEN.

        --------------------------------------------------
        EXAMPLE 1

        chunk_pre:
        Apple Inc. is an American technology company.

        chunk_target:
        It was founded in 1976. The company designs consumer electronics.


        OUTPUT:
        {{
          "resolved_chunk": "Apple Inc. was founded in 1976. The company designs consumer electronics."
        }}

        (Explanation: "It" was replaced with "Apple Inc.". No other text was changed.)
        --------------------------------------------------
        EXAMPLE 2

        chunk_pre:
        Alan Turing was a British mathematician and computer scientist.

        chunk_target:
        He later proposed a method to evaluate machine intelligence, which is now known as the Turing Test.


        OUTPUT:
        {{
          "resolved_chunk": "Alan Turing later proposed a method to evaluate machine intelligence, which is now known as the Turing Test."
        }}

        (Explanation: "He" was replaced with "Alan Turing". No other text was changed.)
"""

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
prompt_arr=[]
for value in chunk_results[0]:
    if value != 0:
        prompt=template.format(chunk_pre=chunk_inf[1][value-1],chunk_target=chunk_inf[1][value])
        prompt_arr.append(prompt)


result=llm.generate_all(
    system_prompt="You are performing coreference resolution for knowledge graph construction.",
    prompt_list=prompt_arr)
print(result)
for index,value in enumerate(result):
    obj = json.loads(value)  # str → dict
    print(obj["resolved_chunk"])
    chunk_inf[1][chunk_results[0][index]]=obj["resolved_chunk"]
print(chunk_inf[1])
for value in chunk_inf[1]:
    print(value)




