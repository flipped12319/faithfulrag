from typing import List

from prompt import PromptGenerator
from qwen import LLMWithThinking
from datasets import load_dataset,Dataset
import re
from qwenBatch import test
import torch
from util.format_util import FormatConverter

model_name="Qwen/Qwen3-8B"
# model=LLMWithThinking("Qwen/Qwen3-8B")
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

    async def generate_knowledge(
            self,
            dataset,
    ):

        prompts = [
            self.prompt_generator_extract.generate_factual_knowledge(
                user_query=item['question']
            )
            for item in dataset
        ]
        # Generate responses using LLM backend
        # merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results =await test(prompts,self.model,self.tokenizer)
        # Parse and return facts
        return [
            {
                'id': item['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for item, result in zip(dataset, raw_results)
        ]

    async def generate_self_context(
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
        raw_results = await test(prompts,self.model,self.tokenizer)
        # Return contexts
        return [
            {'id': item['id'], 'context': result}
            for item, result in zip(dataset, raw_results)
        ]

    async def extract_facts(
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
        prompts = [
            self.prompt_generator_extract.generate_context_extract(
                user_context=ctx['context']
            )
            for ctx in contexts
        ]

        # Generate responses using LLM backend
        raw_results = await test(
            prompts=prompts,
            model=self.model,
            tokenizer=self.tokenizer
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
    def __init__(self,model,tokenizer):
        self.embedding_model = model
        self.tokenizer=tokenizer

    def chunk_text(self,paragraph: str, chunk_size: int = 20) -> List[str]:
        sentences = self.tokenizer.tokenize(paragraph)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def calculate_similarity(
            self,
            paragraph: str,
            str_list: List[str],
            top_k: int = 5,
            chunk_size: int = 50,

    ):
        chunks = self.chunk_text(paragraph, chunk_size=chunk_size)

        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        str_list_embeddings = self.embedding_model.encode(str_list, convert_to_tensor=True, show_progress_bar=False)

        results = []

        for i, string in enumerate(str_list):
            # cosine_scores = util.cos_sim(str_list_embeddings[i], chunk_embeddings)
            similarity = self.embedding_model.similarity(str_list_embeddings[i], chunk_embeddings)
            top_values, top_indices = torch.topk(similarity, k=top_k)
            # top_results = sorted(
            #     enumerate(cosine_scores[0].tolist()), key=lambda x: x[1], reverse=True
            # )[:top_k]
            # print("top_indices: ",top_indices[0].tolist())
            # print("top_values: ", top_values.tolist())
            top_chunks = [(chunks[idx], score) for idx, score in zip(top_indices[0].tolist(),top_values[0].tolist())]
            results.append((string, top_chunks))

        return results

    def get_contextual_chunks(self,facts,dataset,sent_topk=5,chunk_size=20):
        all_chunks = []
        for item,fact in zip(dataset,facts):
            if len(fact['facts']) == 0:
                print('No facts found')
                continue
            paragraph = FormatConverter.remove_brackets_and_content(item['context'])
            results = self.calculate_similarity(paragraph, fact['facts'], top_k=sent_topk, chunk_size=chunk_size)
            chunks = []
            for _,match in results:
                for chunk,score in match:
                    chunk_text = chunk.replace("Ġ", " ")
                    chunk_text = " ".join(chunk_text.split())
                    chunks.append({'chunk':chunk_text,'score':score})

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
            prompts = []
            for item in dataset:
                prompts.append(
                    self.prompt_generator_qa.generate_qa_prompt_without_fact(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                    )
                )
            raw_results = await test(
                prompts=prompts,
                model=self.model,
                tokenizer=self.tokenizer
            )

            # Return predictions
            return {item['id']: res for item, res in zip(dataset, raw_results)}

        async def predict_answer_normal_cot(
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
                    self.prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
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
