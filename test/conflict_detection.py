from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

text="The Laleli Mosque was built in the 20th century"
prompt="""
You are a factual verifier 

Given a single factual statement, determine whether it conflicts with well-established world knowledge
based on your own internal knowledge (not external sources).

Statement:
"{claim}"

Output: {{
    "claim":"{claim}",
    "has_conflict":"...",
    "Reason":"...",
    "internal_knowledge":"...",
  
}}
Rules:
Output JSON only
- Use double quotes (") for all strings and keys.
- Do NOT use single quotes.
About has_conflict in Output
- Use "yes" if the statement contradicts well-established world knowledge.
- Use "no" if the statement does not contradict well-established world knowledge.
- Use "uncertain" only if you are not confident about the conflict or you don't have the internal_knowledge about the claim
"""
type1="numerical"
type2="Temporal"
type3="Attribute"
prompt=prompt.format(claim=text)
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
        {"role": "system", "content": "You are a conflict detector in a Retrieval-Augmented Generation (RAG) system."},
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
