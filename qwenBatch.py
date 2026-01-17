from qwen_instruct import LLMWithThinking
import asyncio
#
# async def batch_generate(model, prompts):
#     """
#     异步批量生成模型输出
#     """
#     # 创建异步任务列表
#     tasks = [model.generate_async(prompt) for prompt in prompts]
#
#     # 并发执行所有任务
#     results = await asyncio.gather(*tasks)
#
#     return results
async def batch_generate(model,system_prompt,prompts, max_concurrent=1):
    """
    异步批量生成模型输出，限制同时运行的任务数量
    """
    semaphore = asyncio.Semaphore(max_concurrent)  # 限制最大并发数

    async def safe_generate(system_prompt_inside,prompt):
        async with semaphore:  # 同时只能进入 max_concurrent 个任务
            return await model.generate_async(system_prompt_inside,prompt)

    tasks = [safe_generate(system_prompt,prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results


# 运行


async def test(system_prompt,prompts,model,tokenizer):
    model = LLMWithThinking(model,tokenizer)
    results = await batch_generate(model,system_prompt,prompts)
    # print("results ",results)
    return results
    # contexts = [item["content"] for item in results]
    # return contexts