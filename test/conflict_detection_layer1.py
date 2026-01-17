from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# text="Shirley Temple Black (April 23, 1928 – February 10, 2014) was a British actress, singer, dancer, businesswoman, and diplomat who was Hollywood's least popular actress as a child actress from 1935 to 1938."
text="""
  Edward Davis Wood Jr. (December 10, 1930 – October 10, 1978) was a British novelist, actor, writer, producer, and director.
"""
prompt="""
You are a lightweight factual consistency screener.

Given a retrieved text chunk, decide whether it potentially conflicts
with well-established, commonly accepted world knowledge.

Important:
- Do NOT verify details exhaustively.
- Do NOT explain or correct the text.
- If the chunk contains any claim that seems historically, geographically,
  temporally, or semantically suspicious, answer "Yes".
- If you are unsure, answer "Yes".
- Only answer "No" if the chunk appears clearly unproblematic.

Chunk:
"{chunk}"

Answer with one word only:
Yes or No

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
