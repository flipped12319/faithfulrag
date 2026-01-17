from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
template="""
    You are an information extraction system.
    
    ### Input
    You are given a text chunk. Coreference has already been resolved.
    All pronouns refer to explicit entities.
    
    chunk:
    {chunk}
    
    ### Task
    Extract factual knowledge as atomic triples.
    
    Each triple must represent exactly ONE fact.
    
    ### Extraction Rules (STRICT)
    1. Each triple must be in the form:
       (subject, relation, object)
    
    2. The subject must be an explicit entity mentioned in the chunk.
    
    3. Do NOT combine multiple facts into one triple.
       - Conjunctions, lists, or parentheses usually imply multiple triples.
    
    4. If a sentence mentions multiple locations, times, or attributes,
       extract one triple per location / time / attribute.
    
    5. Parenthetical information (e.g. dates, locations, explanations)
       should be extracted as separate factual triples when meaningful.
    
    6. Do NOT invent information that is not explicitly stated in the chunk.
    
    7. Do NOT include explanations, summaries, or paraphrases.
    
    8.Do NOT include multiple facts (e.g. time + location) in a single object.
      If both are present, extract them as separate atomic triples.
      
    
    ### Output Format (JSON ONLY)
    Return a JSON object with a single field "triples".
    
    Each triple must be an object with:
    - subject
    - relation
    - object
    
    Example output format:
    {{
      "triples": [
        {{
          "subject": "...",
          "relation": "...",
          "object": "..."，
        }}
      ]
    }}
    
    ### Important Constraints
    - Use concise relation phrases (verb or verb_phrase).
    - One triple = one atomic fact.
    - Output ONLY valid JSON. Do not include any extra text.

"""
# template="""
#         You are an information extraction assistant.
#
#         Given a text, extract factual knowledge triples in JSON format.
#
#         Rules:
#         - Each triple must be (subject, relation, object).
#         - Use atomic, normalized relations (e.g., spouse, profession, award).
#         - Do NOT include relationship meaning inside the object.
#         - Split combined facts into multiple triples when necessary.
#         - Use entities as subjects/objects, not descriptive phrases.
#
#         Output JSON only, following this schema:
#
#         {{
#           "triples": [
#             {{
#               "subject": "...",
#               "relation": "...",
#               "object": "...",
#             }}
#             ...
#           ]
#         }}
#
#         ### Sentences
#         {SENTENCES}
# """
context="This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels)"
prompt=template.format(chunk=context)
# ========== 1. 批量 messages ==========
messages_list = [
    [
        {"role": "system", "content": "You are an information extraction assistant."},
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
