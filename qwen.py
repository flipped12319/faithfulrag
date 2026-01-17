import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer



class LLMWithThinking:
    def __init__(self, model,tokenizer):

        self.model=model
        self.tokenizer = tokenizer
        self.device = "cuda"

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> dict:
        """
        生成模型输出，并分离思考内容（<think>）和最终回答。
        """
        messages = [{"role": "user", "content": prompt}]

        # 构造输入模板（支持“思考模式”）
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # 生成结果
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        # 提取输出部分
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 找到 </think> 结束标记（token id 151668）
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # 分别解码
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # return {
        #     "thinking": thinking_content,
        #     "content": content
        # }

        return {
            "content": content
        }
    async def generate_async(self, prompt: str, max_new_tokens: int = 1024) -> dict:
        """
        异步版本 generate 方法，使用线程池执行同步函数
        """
        return await asyncio.to_thread(self.generate, prompt, max_new_tokens)
