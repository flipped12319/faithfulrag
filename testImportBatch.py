import asyncio
from qwenBatch import test
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name="Qwen/Qwen3-8B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
prompts = [
    "Explain the theory of relativity in simple terms.explain in ten words",
    "Who invented the light bulb?explain in ten words",
    "Describe photosynthesis.explain in ten words"
]
results = asyncio.run(test(prompts,model,tokenizer))

# results=test(prompts)
print(results)
# contexts = [item["content"] for item in results]
# print(type(results))
# print(contexts)