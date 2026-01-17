import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer



class LLMWithThinking:
    def __init__(self, model,tokenizer):

        self.model=model
        self.tokenizer = tokenizer
        self.device = "cuda"

    def generate(self,system_prompt,prompt: str, max_new_tokens: int = 512) -> dict:
        """
        生成模型输出，并分离思考内容（<think>）和最终回答。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # 构造输入模板（支持“思考模式”）
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            padding=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            use_cache=True,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    async def generate_async(self,system_prompt,prompt: str, max_new_tokens: int = 512) -> dict:
        """
        异步版本 generate 方法，使用线程池执行同步函数
        """
        return await asyncio.to_thread(self.generate,system_prompt,prompt, max_new_tokens)
