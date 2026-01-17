from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

text="""
 Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.He lives in Canada"
"""
question="What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
prompt="""
You are an information extraction system.

Extract all atomic, verifiable factual claims from the following text.
Each claim should be a single, standalone fact that can be independently verified.

Rules:
- Do NOT infer or add new information.
- Do NOT merge multiple facts into one claim.
- Use simple declarative sentences.
- Focus on entities, dates, locations, roles, and attributes.

Text:
{chunk}

Output (JSON array only):
[
  "claim 1",
  "claim 2",
  ...
]
"""

prompt=prompt.format(chunk=text)
print(prompt)
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# ========== 1. 批量 messages ==========
messages_list = [
    [
        {"role": "system", "content": "You are an information extraction system."},
        {"role": "user", "content": prompt}
    ],
]

# ========== 2. 批量转模板文本 ==========
batch_texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    for messages in messages_list
]

# ========== 3. 批量 Tokenize ==========
model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)

# ========== 4. 批量生成 ==========
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=512
)

# ========== 5. 截断 prompt ==========
outputs = []
for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
    outputs.append(output_ids[len(input_ids):])

# ========== 6. 批量解码 ==========
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# print all
for i, res in enumerate(responses):
    print(f"=== sample {i} ===")
    print(res)
