from datasets import load_dataset
from prompt import PromptGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from qwen_instruct import LLMWithThinking
import json
from format_util import FormatConverter
from difflib import SequenceMatcher
import random
import json



# Login using e.g. `huggingface-cli login` to access this dataset

# # load the QA pairs
# llm_model_name="Qwen/Qwen2-7B-Instruct"
# # model_name="Qwen/Qwen3-8B-FP8"
# device = "cuda"
# llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# llm_model = AutoModelForCausalLM.from_pretrained(
#             llm_model_name,
#             torch_dtype="auto",
#             device_map="cuda"
#         )
# llm=LLMWithThinking(model=llm_model,tokenizer=llm_tokenizer)
# # system_prompt="You are a data generator for conflict detection in RAG systems."
# system_prompt="You are a helpful model to generate the true answer"
# prompt=["please explain yourself"]
# res=llm.generate_all(system_prompt=system_prompt,prompt_list=prompt)
# print(res)
# print(ds["validation"])
# print(ds["validation"][0])
# print(ds["validation"][0]["context"]["sentences"])
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
dataArr=[]
for i in range(0,5):
    dataArr.append(ds["validation"][i])
# for i,data in enumerate(dataArr):
#     text = []
#     print(i)
#     for sentences in dataArr[i]["context"]["sentences"]:
#         for sentence in sentences:
#             # print(sentence)
#             text.append(sentence)
#     print(text)
#     dataArr[i]["context"]=text
#     print(dataArr[i])

for i, data in enumerate(dataArr):
    text = ""
    print(i)
    for sentences in dataArr[i]["context"]["sentences"]:
        for sentence in sentences:
            text += sentence + " "   # 正确写法
    text = text.strip()  # 去掉末尾空格
    print(text)
    dataArr[i]["context"] = text
    print(dataArr[i])


print(dataArr)
#
# prompt_generator=PromptGenerator()
# res=prompt_generator.generate_conflict_prompt(conflictType="numerical",context="hello")
# print(res)