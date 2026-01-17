from module import FactMiningModule,ContextualAlignmentModule,SelfThinkModule
# from module import FactMiningModule,SelfThinkModule
# from new_module import ContextualAlignmentModule
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import json
from format_util import FormatConverter
from difflib import SequenceMatcher
from prompt import PromptGenerator
from qwen_instruct import LLMWithThinking
import random
import re
# from datasets import Dataset

def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

llm_model_name="Qwen/Qwen2-7B-Instruct"
# model_name="Qwen/Qwen3-8B-FP8"
device = "cuda"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
fact_mining_module=FactMiningModule(llm_model,llm_tokenizer)
llm=LLMWithThinking(model=llm_model,tokenizer=llm_tokenizer)

# dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0")
# def flatten_choices(example):
#     # 提取 choices["text"]
#     return {"choices": example["choices"]["text"]}
#
# dataset = dataset.map(flatten_choices)

# dataset = load_dataset("G20-CS4248/squad1.1")

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

dataArr=[]
# data_num=1
# random_numbers = [random.randint(0, 900) for _ in range(data_num)]
# for i in random_numbers:
#     data.append(dataset["test"][i])
# for i in range(0,1):
#     data.append(dataset["test"][i])
# data.append(dataset["validation"][1])
# print(dataset["test"][1])


#generating conflicts
# prompt_generator=PromptGenerator()
# newDataArr=[]
# indexArr=[]
# for i in range(0,3):
#     newDataArr.append(dataset["validation"][i])
# for index,value in enumerate(newDataArr):
#     title=newDataArr[index]["context"]["title"]
#     hint=newDataArr[index]["supporting_facts"]["title"]
#     tempArr=[]
#     for title_index,title_value in enumerate(title):
#         if title_value in hint:
#             tempArr.append(title_index)
#     indexArr.append(tempArr)
#     print("context",title)
#     print("hint", hint)
#     print("indexArr", indexArr)
# system_prompt="You are a data generator for conflict detection in RAG systems."
# for index,value in enumerate(newDataArr):
#     sentences=newDataArr[index]["context"]["sentences"]
#     arr=indexArr[index]
#     promptArr=[]
#     for i in arr:
#         prompt=prompt_generator.generate_conflict_prompt(conflictType="attribute",context=sentences[i])
#         promptArr.append(prompt)
#     res_json=llm.generate_all(system_prompt=system_prompt,prompt_list=promptArr)
#     print(res_json)
#     for res_index,res_value in enumerate(res_json):
#         res_value = res_value.strip()
#         res_value = re.sub(r"^```json\s*", "", res_value)
#         print(res_value)
#         output = json.loads(res_value)
#         tempArr=[]
#         tempArr.append(output["conflicts"]["text"])
#         newDataArr[index]["context"]["sentences"][arr[res_index]]=tempArr
#         print(output["conflicts"]["text"])
#         print(newDataArr[index]["context"]["sentences"])
# for data in newDataArr:
#     dataArr.append(data)

# 针对hotpotqa数据集进行优化
for i in range(0,10):
    dataArr.append(dataset["validation"][i])



for i, data in enumerate(dataArr):
    text = ""
    # print(i)
    for sentences in dataArr[i]["context"]["sentences"]:
        for sentence in sentences:
            text += sentence + " "   # 正确写法
    text = text.strip()  # 去掉末尾空格

    dataArr[i]["context"] = text

print(dataArr)


embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
# contextual_alignment_module=ContextualAlignmentModule(embedding_model,embedding_tokenizer,llm_model,llm_tokenizer)
contextual_alignment_module=ContextualAlignmentModule(embedding_model,embedding_tokenizer)
self_think_module=SelfThinkModule(llm_model,llm_tokenizer)

#fact generation
formatUtil=FormatConverter()
knowledge = fact_mining_module.generate_knowledge(dataArr)
results=fact_mining_module.generate_self_context(dataArr,knowledge)
facts=fact_mining_module.extract_facts(results)

print(knowledge)
print(results)
print(facts)


print("knowledge len",len(knowledge))
for value in knowledge:
    print(value)

print("results len",len(results))
for value in results:
    print(value)
print("facts len",len(facts))
for value in facts:
    print(value)

#contextual alignment
# chunks=contextual_alignment_module.chunk_text(dataArr[0]["context"])
# print("chunks ",chunks)
# chunks_arr=chunks[1]
# print(chunks_arr)
# for index,value in enumerate(chunks_arr):
#     print(chunks_arr[index])

# contextual_chunks=contextual_alignment_module.get_contextual_chunks(facts,dataArr)
# print(contextual_chunks)
# contextual_chunks_topk=contextual_alignment_module.get_topk_contextual_chunks(contextual_chunks,5)
# print(contextual_chunks_topk)
# print("len",len(contextual_chunks_topk))
# for value in contextual_chunks_topk:
#     print(value)
#
# prompt_generator=PromptGenerator()
#
    # print(len(res))
    # claims = json.loads(res[0])
    # print("claim",claims[0])

# predict_answer_normal=self_think_module.predict_answer_normal_cot(dataArr,contextual_chunks_topk)
# print(predict_answer_normal)
# #
# total_num_f=0
# ex_acc=0
# for answer,item in zip(predict_answer_normal.values(),dataArr):
#         print("\nfaithful")
#         print('item; ',item["id"])
#         print('item answer; ', item["answer"])
#         # print("item options", item["choices"])
#         print('raw answer: ', answer)
#         total_num_f=total_num_f+1
#         answer_dict = json.loads(answer)
#         print("answer_dict ",answer_dict)







        # clean_result=formatUtil.normalize_answer(answer_dict["Answer"])
        # clean_answer = formatUtil.normalize_answer(item["answer"])
        # if clean_answer==clean_result:
        #     ex_acc=ex_acc+1
        # print("faithful ex accuracy: ", ex_acc/total_num_f)
