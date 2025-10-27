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
            enable_thinking=False
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
    async def generate_async(self, prompt: str, max_new_tokens: int = 2048) -> dict:
        """
        异步版本 generate 方法，使用线程池执行同步函数
        """
        return await asyncio.to_thread(self.generate, prompt, max_new_tokens)
# load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="cuda"
# )

# prepare the model input
# prompt = """
#         Task Description:
#         You are an expert in problem analysis. When a user presents a question, your task is to identify the factual knowledge required to answer the question.
#         Please list the relevant facts in a clear and structured manner.
#
#         Instructions:
#             Analyze the question carefully.
#             Identify key areas of knowledge that are crucial for answering the question.
#             Provide a brief explanation of why each area is necessary and follow the Example:
#
#         Question:
#         Who invented the theory of general relativity?
#
#         Answer:
#         To answer this question, the following areas of knowledge are required:
#             1. The theory of general relativity describes gravity as a curvature of spacetime caused by mass and energy.
#             2. The theory was developed by Albert Einstein in 1915.
#             3. Albert Einstein is a German-born theoretical physicist who also developed the equation E = mc², which expresses the equivalence of energy and mass.
#
#         Now, please analyze the following question:
#
#         Question:
#         An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total distance of 2.0 m vertically. Which of the following best describes the objects after they have fallen a distance of 1.0 m?
#
#         Answer:
#             1. your first explanation
#             2. your second explanation
#             3. continue as needed
#         """
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#
# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
#
# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0
#
# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#
# print("thinking content:", thinking_content)
# print("content:", content)