from qwen import LLMWithThinking
import asyncio

async def batch_generate(model, prompts):
    """
    异步批量生成模型输出
    """
    # 创建异步任务列表
    tasks = [model.generate_async(prompt) for prompt in prompts]

    # 并发执行所有任务
    results = await asyncio.gather(*tasks)

    return results


# 使用示例
async def main(prompts,model,tokenizer):
    model = LLMWithThinking(model,tokenizer) # 你的模型实例

    results = await batch_generate(model,prompts)

    for i, res in enumerate(results):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Output: {res['content']}")
        print("-" * 50)
    return results

# 运行


async def test(prompts,model,tokenizer):
    model = LLMWithThinking(model,tokenizer)
    results = await batch_generate(model, prompts)
    contexts = [item["content"] for item in results]
    # for i, res in enumerate(results):
    #     print(f"Prompt {i+1}: {prompts[i]}")
    #     print(f"Output: {res['content']}")
    #     print("-" * 50)
    return contexts