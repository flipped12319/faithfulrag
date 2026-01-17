from typing import List

from prompt import PromptGenerator

import re
import torch
from format_util import FormatConverter
from qwen_instruct import LLMWithThinking
from sentence_transformers import SentenceTransformer
import json
import json
from sympy import false
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk

import spacy
nlp = spacy.load("en_core_web_sm")
def embed_texts(texts, model):
    """批量 embedding，返回 numpy 数组 (N, D)"""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(emb_a, emb_b,model) -> float:
    return model.similarity(emb_a,emb_b)

def triplet_similarity(
    tA,
    tB,
    model,
    w_head: float = 0.4,
    w_rel: float = 0.2,
    w_tail: float = 0.4,
) -> float:
    """
    对两个三元组进行相似度计算：
        head vs head
        relation vs relation
        tail vs tail
    最终用权重加权融合。
    """

    # 分别嵌入三个部分
    embs = embed_texts(
        [tA["head"], tA["relation"], tA["tail"],
         tB["head"], tB["relation"], tB["tail"]],
        model
    )
    hA, rA, tA_emb = embs[0], embs[1], embs[2]
    hB, rB, tB_emb = embs[3], embs[4], embs[5]

    # 计算三部分相似度
    sim_head = cosine_sim(hA, hB,model)
    sim_rel  = cosine_sim(rA, rB,model)
    sim_tail = cosine_sim(tA_emb, tB_emb,model)

    # 加权融合
    final_sim = w_head * sim_head + w_rel * sim_rel + w_tail * sim_tail
    return float(final_sim)

def most_similar_triple(tripleA,tripleArr,model):
    max_index=0
    max_similarity=0
    for index,triple in enumerate(tripleArr):
        similarity=triplet_similarity(tripleA,triple,model)
        # print("index",index)
        # print("similarity",similarity)
        if similarity>max_similarity:
            max_similarity=similarity
            max_index=index
            # print("max_index",max_index)
    return max_index

NEGATION_PATTERNS = [
    "not",
    "n't",
    "no",
    "never",
    "none",
    "cannot",
    "can't",
    "does not",
    "do not",
    "did not",
    "is not",
    "was not",
    "are not",
    "were not",
    "has not",
    "have not",
    "had not",
    "cannot",
    "without",
]
#判断text里是否含有否定词
def has_negation(text: str) -> bool:
    text = text.lower().strip()
    for pat in NEGATION_PATTERNS:
        if pat in text:
            return True
    return False

def load_antonym_lexicon(path):
    antonym_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            cols = line.split(",")

            if len(cols) < 3:
                continue

            word = cols[0].strip().lower()

            # 第二列是同义词，不用
            antonyms_raw = cols[2].strip()

            # 反义词用 # 分隔
            antonyms = [a.strip().lower() for a in antonyms_raw.split("#") if a.strip()]

            # 保存到 dict
            if antonyms:
                antonym_dict[word] = antonyms

    return antonym_dict
dict_final=load_antonym_lexicon("syn-ant.csv")

#将text转变为词元的数组
def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

#判断两个三元组里是否存在冲突
def single_conflict_detection(tA,tB,model,dict_final):
    relation_conflict_flag = False
    negation_conflict_flag=False
    embs = embed_texts(
        [tA["head"], tA["relation"], tA["tail"],
         tB["head"], tB["relation"], tB["tail"]],
        model
    )
    hA, rA, tA_emb = embs[0], embs[1], embs[2]
    hB, rB, tB_emb = embs[3], embs[4], embs[5]

    #calculate similarity
    sim_head = cosine_sim(hA, hB,model)
    sim_rel  = cosine_sim(rA, rB,model)
    sim_tail = cosine_sim(tA_emb, tB_emb,model)


    # relation上是否有冲突

    flag_a = has_negation(tA["relation"])
    flag_b = has_negation(tB["relation"])
    if flag_a != flag_b:
        negation_conflict_flag=True

    r1_words = lemmatize(tA["relation"])
    r2_words = lemmatize(tB["relation"])
    for w1 in r1_words:
        for w2 in r2_words:
            if w1 in dict_final and w2 in dict_final[w1]:
                relation_conflict_flag=True
            if w2 in dict_final and w1 in dict_final[w2]:
                relation_conflict_flag = True

    # head,tail高度相似的情况
    if sim_head>0.75 and sim_tail>0.75:
        if relation_conflict_flag!=negation_conflict_flag:
            return True
    # head和relation高度
    elif sim_head>0.75 and sim_rel>0.8 and (relation_conflict_flag==negation_conflict_flag):
        if sim_tail<0.45:
            return True
    elif sim_tail>0.75 and sim_rel>0.8 and (relation_conflict_flag==negation_conflict_flag):
        if sim_head<0.45:
            return True

    return false

#获取两个多三元组的冲突检测得分
def group_conflict_detection_score(triple_group_A,triple_group_B,model,dict_final):
    score=0
    for triple in triple_group_A:
        index=most_similar_triple(triple,triple_group_B,model)
        print("triple",triple)
        print("index,",index)
        print(single_conflict_detection(triple,triple_group_B[index],model, dict_final))
        if single_conflict_detection(triple,triple_group_B[index],model, dict_final):
            score+=1

    return score
class FactMiningModule:
    def __init__(self,model,tokenizer):
        self.prompt_generator = PromptGenerator(
            task="normal"
        )
        self.prompt_generator_extract = PromptGenerator(
            task="extract"
        )
        self.fact_pattern = r'\d+\.\s([^\d]+(?:\s+[^\d]+)*)'
        # self.model=LLMWithThinking(model_name,tokenizer)
        self.model=model
        self.tokenizer=tokenizer

    def generate_knowledge(self, dataset):
        llm = LLMWithThinking(self.model, self.tokenizer)
        prompts = [
            self.prompt_generator_extract.generate_factual_knowledge(
                user_query=item['question']
            )
            for item in dataset
        ]
        system_prompt="You are a helpful,respectful and honest assistant. Always answer as helpfully as possible"
        raw_results = llm.generate_all(
            system_prompt=system_prompt,
            prompt_list=prompts,
            batch_size=3
        )
        return [
            {
                'id': item['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for item, result in zip(dataset, raw_results)
        ]

    def generate_self_context(
            self,
            dataset,
            # knowledges: Optional[Union[Dict, List[Dict]]] = None,
            knowledges,
            **sampling_kwargs
    ):
        """
        Generate self-context for each item in the dataset

        Args:
            dataset: Input dataset
            knowledges: Optional factual knowledge to incorporate
            sampling_kwargs: Generation parameters to override defaults

        Returns:
            List of dictionaries containing context for each item
        """
        # Generate prompts for context generation
        llm = LLMWithThinking(self.model, self.tokenizer)
        prompts = []
        for item in dataset:
            if knowledges is None:
                prompt = self.prompt_generator.generate_context_directly_prompt(
                    user_query=item['question']
                )
            else:
                # Find matching knowledge for this item
                knowledge = next(
                    (k for k in knowledges if k['id'] == item['id']),
                    None
                )
                if knowledge is None:
                    # logger.warning(f"No knowledge found for item {item['id']}")
                    prompt = self.prompt_generator.generate_context_directly_prompt(
                        user_query=item['question']
                    )
                else:
                    prompt = self.prompt_generator.generate_context_by_factual_knowledge(
                        user_query=item['question'],
                        factual_knowledge=knowledge['facts']
                    )
            prompts.append(prompt)

        # Generate responses using LLM backend
        # merged_params = {**self.default_sampling_params, **sampling_kwargs}
        # logger.info(f"Generating self-contexts...")
        system_prompt = "You are a helpful,respectful and honest assistant. Always answer as helpfully as possible"
        raw_results = llm.generate_all(
            system_prompt=system_prompt,
            prompt_list=prompts,
            batch_size=3
        )
        # Return contexts
        return [
            {'id': item['id'], 'context': result}
            for item, result in zip(dataset, raw_results)
        ]

    def extract_facts(
            self,
            contexts,
    ):
        """
        Extract facts from given contexts

        Args:
            contexts: List of contexts or single context dictionary
            sampling_kwargs: Generation parameters to override defaults

        Returns:
            List of dictionaries containing extracted facts
        """
        # Normalize input to list
        if isinstance(contexts, dict):
            contexts = [contexts]

        # Generate prompts for fact extraction
        llm = LLMWithThinking(self.model, self.tokenizer)
        prompts = [
            self.prompt_generator_extract.generate_context_extract(
                user_context=ctx['context']
            )
            for ctx in contexts
        ]
        system_prompt="You are a precise and reliable information extractor.Your sole task is to extract relevant information from the given context strictly according to the instructions.You must not add, modify,or infer any information that is not explicitly stated in the context"
        # Generate responses using LLM backend
        raw_results = llm.generate_all(
            system_prompt=system_prompt,
            prompt_list=prompts,
            batch_size=3
        )
        # Parse and return facts
        return [
            {
                'id': ctx['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for ctx, result in zip(contexts, raw_results)
        ]


class ContextualAlignmentModule:
    def __init__(self,model,tokenizer,llm_model,llm_tokenizer):
        self.embedding_model = model
        self.tokenizer=tokenizer
        self.llm_model=llm_model
        self.llm_tokenizer=llm_tokenizer
        self.prompt_generator_extract = PromptGenerator(
            task="extract"
        )

    def chunk_text(self,paragraph: str, chunk_size: int = 20):
        sentences = self.tokenizer.tokenize(paragraph)
        chunk_num=0
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                chunk_num=chunk_num+1
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))
            chunk_num = chunk_num + 1

        chunk_inf=[]
        chunk_inf.append(chunk_num)
        chunk_inf.append(chunks)

        return chunk_inf


    def calculate_similarity(
            self,
            paragraph: str,
            str_list: List[str],
            top_k: int = 10,
            chunk_size: int = 50,

    ):
        chunks_inf = self.chunk_text(paragraph, chunk_size=chunk_size)
        chunks=chunks_inf[1]
        chunk_num=chunks_inf[0]

        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        str_list_embeddings = self.embedding_model.encode(str_list, convert_to_tensor=True, show_progress_bar=False)

        results = []

        for i, string in enumerate(str_list):
            similarity = self.embedding_model.similarity(str_list_embeddings[i], chunk_embeddings)
            if chunk_num<top_k:
                top_values, top_indices = torch.topk(similarity, k=chunk_num)
            else:
                top_values, top_indices = torch.topk(similarity, k=top_k)
            top_chunks = [(chunks[idx], score) for idx, score in zip(top_indices[0].tolist(),top_values[0].tolist())]
            results.append((string, top_chunks))

        return results

    def get_contextual_chunks(self,facts,dataset,sent_topk=10,chunk_size=20):
        llm = LLMWithThinking(self.llm_model, self.llm_tokenizer)
        all_chunks = []
        for item,fact in zip(dataset,facts):
            if len(fact['facts']) == 0:
                print('No facts found')
                continue
            paragraph = FormatConverter.remove_brackets_and_content(item['context'])
            results = self.calculate_similarity(paragraph, fact['facts'], top_k=sent_topk, chunk_size=chunk_size)
            print("results,", results)

            chunks = []
            system_prompt = "You are an expert information extraction system that extracts precise subject–relation–object triples.You must not output any reasoning steps, chain-of-thought, analysis, explanation, or internal thinking. Do NOT output anything except the final JSON result."
            for sin_fact,match in results:
                print("fact",sin_fact)
                prompt_arr=[]
                prompt = self.prompt_generator_extract.extract_tripple(sin_fact)
                prompt_arr.append(prompt)

                # print("sin_fact",prompt)
                raw_results = llm.generate_all(
                    system_prompt=system_prompt,
                    prompt_list=prompt_arr,
                    batch_size=1
                )
                fact_triple = json.loads(raw_results[0])
                # fact_triple=raw_results[0]
                print("fact triple",raw_results[0])

                for chunk,score in match:
                    chunk_text = chunk.replace("Ġ", " ")
                    chunk_text = " ".join(chunk_text.split())
                    prompt_arr = []
                    prompt = self.prompt_generator_extract.extract_tripple(chunk_text)
                    prompt_arr.append(prompt)
                    raw_results = llm.generate_all(
                        system_prompt=system_prompt,
                        prompt_list=prompt_arr,
                        batch_size=1
                    )
                    chunk_triple = json.loads(raw_results[0])
                    # chunk_triple=raw_results[0]
                    # print("chunk_text",chunk_text)
                    # print("chunk_triple",raw_results)
                    chunks.append({'chunk':chunk_text,'score':score})
                    # print("chunk_triple",chunk_triple)
                    # print("fact_triple",fact_triple)
                    conflict_score=group_conflict_detection_score(fact_triple,chunk_triple,self.embedding_model,dict_final)
                    print("conflict_score",conflict_score)
                    # flag=single_conflict_detection(tA=fact_triple[0],tB=chunk_triple[0],model=self.embedding_model,dict_final=dict_final)
                    # if flag:
                    #     print("存在冲突")


            all_chunks.append({'id':fact['id'],'chunks':chunks})
        return all_chunks

    def get_topk_contextual_chunks(self, all_chunks, chunk_topk=5):
        all_topk_chunks = []
        for chunk in all_chunks:
            topk_chunks = []
            sorted_chunks = sorted(chunk['chunks'], key=lambda x: x['score'], reverse=True)
            seen_chunks = set()
            # pick unique chunks
            for sub_chunk in sorted_chunks:
                if sub_chunk['chunk'] not in seen_chunks:
                    topk_chunks.append(sub_chunk)
                    seen_chunks.add(sub_chunk['chunk'])
                if len(topk_chunks) == chunk_topk:
                    break
            all_topk_chunks.append({'id': chunk['id'], 'topk_chunks': topk_chunks})
        return all_topk_chunks

class SelfThinkModule:
        """Self-thinking module for answer prediction with multiple LLM backends"""

        def __init__(
            self,
            model,
            tokenizer
        ):
            self.tokenizer = tokenizer
            self.model=model
            self.prompt_generator_qa_cot = PromptGenerator(
                task="qa-cot"
            )
            self.prompt_generator_qa = PromptGenerator(
                task="qa"
            )
        async def predict_answer_without_fact(
                self,
                dataset,
        ):
            llm=LLMWithThinking(self.model,self.tokenizer)
            prompts = []
            for item in dataset:
                prompts.append(
                    self.prompt_generator_qa.generate_qa_prompt_without_fact(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                    )
                )
            system_prompt_cot="You are an expert in retrieval QA and Chain of Thought reasoning.Provide your reasoning steps followed by a precise and direct answer.Avoiding any unnecessary explanations or verbosity"
            system_prompt_wo_cot="You are an expert in retrieval QA.Please respond with the exact answer only.Don't be verbose or provide extra information"
            # raw_results = await test(
            #     system_prompt=system_prompt_wo_cot,
            #     prompts=prompts,
            #     model=self.model,
            #     tokenizer=self.tokenizer
            # )
            raw_results = llm.generate_all(
                system_prompt=system_prompt_cot,
                prompt_list=prompts,
                batch_size=3
            )

            # Return predictions
            return {item['id']: res for item, res in zip(dataset, raw_results)}

        def predict_answer_normal_cot(
                self,
                dataset,
                facts,
        ):
            llm=LLMWithThinking(self.model,self.tokenizer)
            prompts = []
            for item in dataset:
                # Find matching facts for this item
                fact_str = next(
                    (' '.join([chunk['chunk'] for chunk in d['topk_chunks']])
                     for d in facts if d['id'] == item['id']),
                    None
                )

                prompts.append(
                    self.prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                        facts=fact_str
                    )
                )
            system_prompt_cot="You are an expert in retrieval QA and Chain of Thought reasoning.Provide your reasoning steps followed by a precise and direct answer.Avoiding any unnecessary explanations or verbosity"
            system_prompt_wo_cot="You are an expert in retrieval QA.Please respond with the exact answer only.Don't be verbose or provide extra information"

            raw_results = llm.generate_all(
                system_prompt=system_prompt_wo_cot,
                prompt_list=prompts,
                batch_size=3
            )
            # Return predictions
            return {item['id']: res for item, res in zip(dataset, raw_results)}

        async def predict_answer_scheduled_cot(
                self,
                dataset,
                facts,
        ):
            prompts = []
            for item in dataset:
                # Find matching facts for this item
                fact_str = next(
                    (' '.join([chunk['chunk'] for chunk in d['topk_chunks']])
                     for d in facts if d['id'] == item['id']),
                    None
                )

                prompts.append(
                    self.prompt_generator_qa_cot.generate_qa_prompt_schedule_cot(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                        facts=fact_str
                    )
                )
            # print('prompts; ',prompts)
            raw_results = await test(
                prompts=prompts,
                model=self.model,
                tokenizer=self.tokenizer
            )

            # Return predictions
            return {item['id']: res for item, res in zip(dataset, raw_results)}

        async def predict_answer_wo_cot(
                self,
                dataset,
                facts,
        ):
            prompts = []
            for item in dataset:
                # Find matching facts for this item
                fact_str = next(
                    (' '.join([chunk['chunk'] for chunk in d['topk_chunks']])
                     for d in facts if d['id'] == item['id']),
                    None
                )

                prompts.append(
                    self.prompt_generator_qa_cot.generate_qa_prompt(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                        facts=fact_str
                    )
                )
            # print('prompts; ',prompts)
            raw_results = await test(
                prompts=prompts,
                model=self.model,
                tokenizer=self.tokenizer
            )

            # Return predictions
            return {item['id']: res for item, res in zip(dataset, raw_results)}
