from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

text="""
  Radio City is India's first private FM radio station and was started on 3 July 2001.,It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003),
  It plays Hindi, English and regional songs.,
  It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.,
  Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features,
  The Radio station currently plays a mix of Hindi and Regional music,
  Abraham Thomas is the CEO of the company.
"""

prompt="""
You are a data generator for conflict detection in RAG systems.

Your task:
Given a context and a conflict type, generate ONE conflicted version that contradicts the original.

=====================================================================
STRICT GLOBAL RULES
=====================================================================
1. You MUST NOT modify more than **3 locations**.
2. A “location” means one number/date/value inside ONE sentence.
3. Before generating the conflicted text, you MUST output a list called "modifications" that specifies:
    - (a) sentence index to modify
    - (b) exact original value
    - (c) new contradictory value
4. ONLY the listed locations may be changed.
5. Every other part of the text MUST be copied **verbatim** (identical characters).
6. No rewriting, no paraphrasing, no hidden changes.

=====================================================================
ATTRIBUTE-TYPE SPECIAL RULES  
=====================================================================
If conflict_type = "attribute", you MUST:

A. Identify at least 2–5 candidate “attribute properties” from the context  
   (e.g., nationality, occupation, birthplace, species, color, size, language, organization, role...etc).  
   You MUST list these before deciding which to modify.

B. Choose ONLY from those attribute properties.  
   *You MUST NOT modify any temporal, numeric, or date-related information.*  
   Forbidden modifications (MUST NOT appear):
   - years (1990, 2010, etc.)
   - dates (“in 1995”, “on July 3”)
   - ages (“at 20”, “age 32”)
   - any timeline-related words (“later”, “after”, “before”, “since”)

C. Attribute conflict MUST NOT be implemented by modifying years or time.

D. If a candidate attribute cannot be found, you MUST NOT fallback to temporal conflict.  
   Instead, you must return an error message:
   “ERROR: no attribute fields detectable in context.”

=====================================================================
CONFLICT RULES BY TYPE
=====================================================================
- Numerical conflict: only modify numeric quantities not expressing time.
- Temporal conflict: only modify explicit time expressions.
- Attribute conflict: only modify inherent non-temporal properties.

=====================================================================
OUTPUT FORMAT
=====================================================================
{{
  "original": "...",
  "conflict_type": "...",

  "attribute_candidates": [...],   // Only for attribute type

  "modifications": [
      {{"sentence_id": X, "from": "A", "to": "B"}},
      ...
  ],

  "conflicts": {{
      "text": "..."
  }}
}}

type:
{type}

context:
{your_input_fact}

"""
type1="numerical"
type2="Temporal"
type3="Attribute"
prompt=prompt.format(your_input_fact=text,type=type3)
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
        {"role": "system", "content": "You are a data generator for conflict detection in RAG systems."},
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
