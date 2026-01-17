from module import FactMiningModule,ContextualAlignmentModule,SelfThinkModule
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import json
from util.format_util import FormatConverter
from difflib import SequenceMatcher
from vllm import LLM, SamplingParams
import re
from datasets import Dataset

def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


llm_model_name="Qwen/Qwen3-8B"
# model_name="Qwen/Qwen3-8B-FP8"
device = "cuda"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = LLM(model=llm_model_name,trust_remote_code=True,max_model_len=32000)
# llm_model = AutoModelForCausalLM.from_pretrained(
#             llm_model_name,
#             torch_dtype="auto",
#             device_map="cuda"
#         )
fact_mining_module=FactMiningModule(llm_model,llm_tokenizer)

dataset = load_dataset("eric-xiang/FaithfulRAG-Dataset")
data=[]
for i in range(0,1):
    data.append(dataset["train"][i])
# data.append(dataset["train"][134])


embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu"
)
embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
contextual_alignment_module=ContextualAlignmentModule(embedding_model,embedding_tokenizer)
self_think_module=SelfThinkModule(llm_model,llm_tokenizer)
async def main():

    formatUtil=FormatConverter()
    #生成fact
    # knowledge = await fact_mining_module.generate_knowledge(data)
    # print(knowledge)
    # results=await fact_mining_module.generate_self_context(data,knowledge)
    # print(results)
    # facts=await fact_mining_module.extract_facts(results)
    # print(facts)
    #
    # #contextual alignment
    # contextual_chunks=contextual_alignment_module.get_contextual_chunks(facts,data)
    # print(contextual_chunks)
    # contextual_chunks_topk=contextual_alignment_module.get_topk_contextual_chunks(contextual_chunks,5)
    # print(contextual_chunks_topk)
    #
    # predict_answer_normal=await self_think_module.predict_answer_normal_cot(data,contextual_chunks_topk)
    # print(predict_answer_normal)
    #
    # #evaluation
    # total_num_f=0
    # correct_num_f=0
    # for answer,item in zip(predict_answer_normal.values(),data):
    #     print("faithful\n")
    #     print('raw answer: ', answer)
    #
    #     print('\n')
    #     print('item; ',item["id"])
    #     total_num_f=total_num_f+1
    #
    #     try:
    #         answer_dict = json.loads(answer)
    #     except json.JSONDecodeError:
    #         print("json error happened")
    #         print(answer)
    #         answer_after=formatUtil.extract_last_answer(answer)
    #         print("-----------------\n")
    #         print('answer:')
    #         print(answer_after)
    #         print("-----------------\n")
    #         similarities = [(c, get_similarity(answer_after, c)) for c in item["choices"]]
    #         best_match = max(similarities, key=lambda x: x[1])
    #         print("best match:\n")
    #         print(best_match[0])
    #         print("finished\n")
    #         if best_match[0]==item["answer"]:
    #             correct_num_f = correct_num_f + 1
    #             print("successfully match!!!")
    #         continue
    #
    #
    #     print("result:", answer_dict["Answer"])
    #     print("\nanswer: ", item["answer"])
    #     if answer_dict["Answer"]==item["answer"]:
    #         correct_num_f=correct_num_f+1
    # print("\nfaithfulrag accuracy: ", correct_num_f/total_num_f)
    #
    total_num=0
    correct_num=0
    predict_answer_without_fact=await self_think_module.predict_answer_without_fact(data)

    for answer,item in zip(predict_answer_without_fact.values(),data):
        print("normal\n")
        print('raw answer: ', answer)

        print('\n')
        print('item; ', item["id"])
        total_num=total_num+1

        try:
            answer_dict = json.loads(answer)

        except json.JSONDecodeError:
            print("json error happened")
            print(answer)
            answer_after=formatUtil.extract_last_answer(answer)
            print("-----------------\n")
            print('answer:')
            print(answer_after)
            print("-----------------\n")
            similarities = [(c, get_similarity(answer_after, c)) for c in item["choices"]]
            best_match = max(similarities, key=lambda x: x[1])
            print("best match:\n")
            print(best_match[0])
            print("finished\n")
            if best_match[0]==item["answer"]:
                correct_num = correct_num + 1
                print("successfully match!!!")
            continue


        print("result:", answer_dict["Answer"])
        print("\nanswer: ", item["answer"])
        if answer_dict["Answer"]==item["answer"]:
            correct_num=correct_num+1
    print("\nrag accuracy: ", correct_num/total_num)
    # print("\nfaithfulrag accuracy: ", correct_num_f / total_num_f)


asyncio.run(main())
# explain in ten words