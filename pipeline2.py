from module import FactMiningModule,ContextualAlignmentModule,SelfThinkModule
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import json
from util.format_util import FormatConverter
from util.evaluate import Evaluate
from difflib import SequenceMatcher
import random
import re
from datasets import Dataset

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

# dataset = load_dataset("eric-xiang/FaithfulRAG-Dataset")

dataset = load_dataset("G20-CS4248/squad1.1")
data=[]
data_num=50
random_numbers = [random.randint(0, 8000) for _ in range(data_num)]
for i in random_numbers:
    data.append(dataset["validation"][i])
# for i in range(0,5):
#     data.append(dataset["validation"][i])
# data.append(dataset["train"][2])
# print(dataset["train"][1])

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
contextual_alignment_module=ContextualAlignmentModule(embedding_model,embedding_tokenizer)
self_think_module=SelfThinkModule(llm_model,llm_tokenizer)
async def main():

    formatUtil=FormatConverter()
    evaluate_helper=Evaluate()
    # # #生成fact
    knowledge = await fact_mining_module.generate_knowledge(data)
    print(knowledge)
    results=await fact_mining_module.generate_self_context(data,knowledge)
    print(results)
    facts=await fact_mining_module.extract_facts(results)
    print(facts)

    # contextual alignment
    contextual_chunks=contextual_alignment_module.get_contextual_chunks(facts,data)
    print(contextual_chunks)
    contextual_chunks_topk=contextual_alignment_module.get_topk_contextual_chunks(contextual_chunks,5)
    print(contextual_chunks_topk)

    predict_answer_normal=await self_think_module.predict_answer_normal_cot(data,contextual_chunks_topk)
    print(predict_answer_normal)

    #
    # evaluation
    total_num_f=0
    correct_num_f=0
    em_num_f=0
    acc_num_f=0
    faithful_error = 0
    similarity_f_total = 0
    f_f1_score=0
    for answer,item in zip(predict_answer_normal.values(),data):
        print("\nfaithful")
        print('item; ',item["id"])
        print('item answer; ', item["answers"][0]["text"])
        print('raw answer: ', answer)
    #
        total_num_f=total_num_f+1
        answer_dict = json.loads(answer)
        print("answer_dict ",answer_dict)

        clean_result=formatUtil.normalize_answer(answer_dict["Answer"])
        clean_answer = formatUtil.normalize_answer(item["answers"][0]["text"])
        # print("clean_result ",clean_result)
        # print("clean_answer ", clean_answer)

        result_embedding=embedding_model.encode(clean_result)
        answer_embedding=embedding_model.encode(clean_answer)
        similarity=embedding_model.similarity(result_embedding,answer_embedding)
        similarity_f_total=similarity_f_total+similarity.item()

        if clean_result in clean_answer:
            acc_num_f=acc_num_f+1

        if clean_answer==clean_result:
            em_num_f=em_num_f+1

        f_f1_score=f_f1_score+evaluate_helper.token_level_f1(clean_result,clean_answer)

        print("similarity",similarity.item())
        if similarity.item()>0.8:
            correct_num_f=correct_num_f+1
    print("faithfulrag accuracy: ", correct_num_f / total_num_f)
    print("faithful em accuracy: ", em_num_f/total_num_f)
    print("averagefrag similarity: ", similarity_f_total / total_num_f)
    print("averagefrag f1score: ", f_f1_score / total_num_f)
    print("acc_f score: ", acc_num_f/ total_num_f)


    #
    total_num=0
    correct_num=0
    similarity_total=0
    em_num=0
    predict_answer_without_fact=await self_think_module.predict_answer_without_fact(data)
    f1_score=0
    acc_num=0
    for answer,item in zip(predict_answer_without_fact.values(),data):
        print("\nnormal")
        print('raw answer: ', answer)

        print('item; ', item["id"])
        print('item answer; ', item["answers"][0]["text"])
        answer_dict = json.loads(answer)
        print("answer_dict ",answer_dict)
        #
        clean_result=formatUtil.normalize_answer(answer_dict["Answer"])
        clean_answer = formatUtil.normalize_answer(item["answers"][0]["text"])
        # print("clean_result ",clean_result)
        # print("clean_answer ", clean_answer)
        result_embedding=embedding_model.encode(clean_result)
        answer_embedding=embedding_model.encode(clean_answer)
        similarity=embedding_model.similarity(result_embedding,answer_embedding)
        similarity_total = similarity_total + similarity.item()
        print("similarity ",similarity.item())
        if clean_result in clean_answer:
            acc_num=acc_num+1
        if clean_answer==clean_result:
            em_num=em_num+1
        f1_score = f1_score + evaluate_helper.token_level_f1(clean_result, clean_answer)
        total_num=total_num+1
        if similarity.item()>0.8:
            correct_num=correct_num+1
    print("rag accuracy: ", correct_num / total_num)
    print("faithfulrag accuracy: ", correct_num_f / total_num_f)

    print("rag em accuracy: ", em_num_f / total_num)
    print("faithful em accuracy: ", em_num_f / total_num_f)

    print("average_rag similarity: ", similarity_total / total_num)
    print("average_f_rag similarity: ", similarity_f_total / total_num_f)

    print("averagerag f1score: ", f1_score / total_num)
    print("averagefrag f1score: ", f_f1_score / total_num_f)

    print("acc score: ", acc_num / total_num)
    print("acc_f score: ", acc_num_f / total_num_f)
    #
    #     try:
    #         answer_dict = json.loads(answer)
    #
    #     except json.JSONDecodeError:
    #         print("json error happened")
    #         normal_error=normal_error+1
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
    #         if formatUtil.clean_string(best_match[0]) == formatUtil.clean_string(item["answer"]):
    #             correct_num = correct_num + 1
    #             print("successfully match!!!")
    #
    #         continue
    #
    #
    #     print("result:", answer_dict["Answer"])
    #     print("\nanswer: ", item["answer"])
    #     print('item; ', item["choices"])
    #     if answer_dict["Answer"] is not None:
    #         if formatUtil.clean_string(answer_dict["Answer"])==formatUtil.clean_string(item["answer"]):
    #             correct_num=correct_num+1
    #
    #
    # print("\nrag accuracy: ", correct_num/total_num)
    # print("\nfaithfulrag accuracy: ", correct_num_f / total_num_f)
    # print("faithfulError:   ",faithful_error)
    # print("normalError:   ", normal_error)


asyncio.run(main())
# explain in ten words