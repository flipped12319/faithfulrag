from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_instruct import LLMWithThinking


device = "cuda" # the device to load the model onto
import spacy
import string
import json


import networkx as nx
import re
import nltk
from nltk.stem import WordNetLemmatizer
import inflect

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
inf = inflect.engine()

RELATION_MAP = {
    "is a": "is_a",
    "is an": "is_a",
    "is": "is",
    "are": "is",
    "takes place in": "occurs_in",
    "take place in": "occurs_in",
    "primarily takes place in": "occurs_in",
    "happens in": "occurs_in",
    "located in": "located_in",
    "part of": "part_of",
    "consists of": "consists_of",
    "causes": "causes",
    "results in": "results_in",
}

def extract_complete_sentences(text, min_tokens=5, allow_long_sentence=True):
    text = text.strip()
    split = re.split(r'([.!?])', text)

    # 如果只有一个 block，说明没有句号等终止符
    if allow_long_sentence and len(split) == 1:
        if len(text.split()) >= min_tokens:
            return [text]  # 当作完整句子返回
        else:
            return []

    sentences = []
    for i in range(0, len(split)-1, 2):
        body = split[i].strip()
        punct = split[i+1]
        full = body + punct
        if len(full.split()) >= min_tokens:
            sentences.append(full)

    return sentences

def basic_clean(raw_text):
    if raw_text is None:
        return ""
    raw_text = raw_text.strip()
    raw_text = raw_text.lower()
    raw_text = re.sub(r"\s+", " ", raw_text)
    return raw_text

def clean_entity(entity):
    entity = basic_clean(entity)

    # 去掉冠词 the / a / an
    entity = re.sub(r"^(the|a|an)\s+", "", entity)

    # 词形还原（主要是名词）
    entity = " ".join([lemmatizer.lemmatize(w, pos="n") for w in entity.split()])

    # 变为 snake_case
    entity = entity.replace(" ", "_")

    return entity

def normalize_relation(rel):
    original = rel
    rel = basic_clean(rel)

    # 去掉副词等无用成分
    rel = re.sub(r"\b(primarily|mainly|mostly|often|usually)\b", "", rel).strip()
    rel = re.sub(r"\s+", " ", rel)

    # 如果能直接匹配词典，优先使用词典
    for k, v in RELATION_MAP.items():
        if rel.startswith(k):
            return v

    # 默认：对动词进行 lemmatization
    tokens = rel.split()
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    rel = "_".join(tokens)

    return rel


# --------------------------------------------------
# 4. 三元组清洗主函数
# --------------------------------------------------
def clean_triple(triple):
    head = clean_entity(triple.get("head", ""))
    relation = normalize_relation(triple.get("relation", ""))
    tail = clean_entity(triple.get("tail", ""))

    # 去掉空 triple
    if not head or not relation or not tail:
        return None

    return {"head": head, "relation": relation, "tail": tail}


# --------------------------------------------------
# 5. 批量清洗（给一个列表）
# --------------------------------------------------
def clean_triple_list(triples):
    cleaned = []
    for t in triples:
        ct = clean_triple(t)
        if ct:
            cleaned.append(ct)
    return cleaned

def triples_to_graph(triples):
    """
    将三元组列表构建为 NetworkX 有向图
    """
    G = nx.DiGraph()
    for t in triples:
        head = t["head"]
        tail = t["tail"]
        relation = t["relation"]

        G.add_node(head)
        G.add_node(tail)
        G.add_edge(head, tail, relation=relation)

    return G




model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# prompt = "Give me a short introduction to large language model."
prompt="""
        You are an expert information extraction system that extracts precise subject–relation–object triples.
        Your output must be structurally correct, complete, and fully grounded in the input text.

        ### Extraction Rules

        1. Extract only factual, explicit information stated in the text.
        2. A triple = {{head, relation, tail}}.
        3. DO NOT generate NULL or empty values.
        4. DO NOT guess or hallucinate missing information.
        5. The **relation must be a complete phrase**, including required prepositions.
           - Example: “takes place in”, “is part of”, “is located in”, “results in”.

        6. The **tail should NOT contain prepositions** (in, on, at, to, from…).
           - These should be moved into the relation.
            For Example:
            Correct:
              relation = "takes place in"
              tail = "the leaves of plants"

            Incorrect:
              relation = "takes place in the leaves of"
              tail = "plants"

        7. Normalize entity phrases:
           - Remove determiners (“the”, “a”, “an”) unless needed.
           - Use base form for concept entities.
        8. Keep wording concise but accurate.
        9. Relation MUST be a concise verb phrase (1–3 words), such as "is a", "causes", "occurs in", "located at".
           Do NOT include long clauses in relation.
        10. Each triple must contain exactly one atomic action or relation.
            If the sentence contains multiple verbs/actions, you MUST extract multiple triples.
            Never merge multiple actions into a single relation. This is strictly forbidden.
            Incorrect Example (DO NOT DO THIS):
                {{
                "head": "The analyst",
                "relation": "used analytical software to examine the dataset",
                "tail": "to discover useful patterns in the recorded results"
                }}
            Correct Example (DO THIS):
                [
                  {{
                    "head": "analyst",
                    "relation": "used",
                    "tail": "analytical software"
                  }},
                  {{
                    "head": "analytical software",
                    "relation": "to examine",
                    "tail": "dataset"
                  }},
                  {{
                    "head": "analyst",
                    "relation": "discovered",
                    "tail": "useful patterns"
                  }},
                  {{
                    "head": "dataset",
                    "relation": "recorded",
                    "tail": "results"
                  }}
                ]
        11.- If the sentence is in *passive voice*, the **true semantic agent** (the entity performing the action) MUST be extracted as the **head**, even if it appears after "by".
              Example:
                Text: "The samples were analyzed by the researcher."
                Correct triple:
                  head = "researcher"
                  relation = "analyzed"
                  tail = "samples"
              
            - The grammatical subject of a passive sentence (e.g., “patterns”, “samples”) MUST NOT be used as the head unless it is truly the agent.
            - Rewrite passive constructions into their active logical form **internally** before extracting triples.
              Example:
                "Meaningful patterns were identified by the software."
                Treat as → "The software identified meaningful patterns."
        12. Output an array of triples. Each triple MUST follow this format:
        [
            {{
            "head": "<entity>",
            "relation": "<relation>",
            "tail": "<entity>"
            }}
        ]
        ### Additional Constraints
        - Do not rewrite, paraphrase, or invent content.
        - Preserve all meaning present in the original sentence.
        - If multiple triples exist, output all of them.
        - If a phrase conveys location, cause, purpose, or property, extract it as a triple.
        - Tail must be a complete noun phrase.
        - If the sentence is incomplete, extract all factual triples that can still be reliably inferred from the completed portion of the text.
          Do not discard triples just because the sentence is unfinished.


        ### Now extract triples from the following text:
        {text}
"""

text_a="Lysosomes play a pivotal role in ensuring that these conditions are met, thereby indirectly contributing to protein building within the cell. "
# text_a="The researcher used statistical software to analyze the dataset and identify meaningful patterns in the collected results."
text_b="The analyst used analytical software to examine the dataset and discover useful patterns in the recorded results."
text_c="Meaningful patterns in the dataset were identified after the collected results had been examined with statistical software by the researcher."
input_text_a=extract_complete_sentences(text_a)
input_text_b=extract_complete_sentences(text_b)
input_text_c=extract_complete_sentences(text_c)
print(input_text_a)
print(input_text_b)
print(input_text_c)
real_prompt_a=prompt.format(text=input_text_a)
real_prompt_b=prompt.format(text=input_text_b)
real_prompt_c=prompt.format(text=input_text_c)

prompt_arr=[]
prompt_arr.append(real_prompt_a)
prompt_arr.append(real_prompt_b)
prompt_arr.append(real_prompt_c)
system_prompt_="You are an expert information extraction system that extracts precise subject–relation–object triples.You must not output any reasoning steps, chain-of-thought, analysis, explanation, or internal thinking. Do NOT output anything except the final JSON result."
llm=LLMWithThinking(model=model,tokenizer=tokenizer)
responses=llm.generate_all(system_prompt=system_prompt_,prompt_list=prompt_arr)
new_triple_arr=[]
graph_arr=[]
for index,res in enumerate(responses):
    print(f"res{index}", res)
    triple=json.loads(res)
    # print(triple)
    new_triple=clean_triple_list(triple)
    new_triple_arr.append(new_triple)
    graph = triples_to_graph(new_triple)
    graph_arr.append(graph)


    #
    # print(new_triple)
print(new_triple_arr)
print(graph_arr)
def jaccard_graph_similarity(G1, G2):
    nodes1, nodes2 = set(G1.nodes()), set(G2.nodes())
    edges1 = {(u, v, G1.edges[u, v]["relation"]) for u, v in G1.edges()}
    edges2 = {(u, v, G2.edges[u, v]["relation"]) for u, v in G2.edges()}

    node_sim = len(nodes1 & nodes2) / len(nodes1 | nodes2) if nodes1 | nodes2 else 1.0
    edge_sim = len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 1.0

    return 0.5 * node_sim + 0.5 * edge_sim  # 你可调权重

print(jaccard_graph_similarity(graph_arr[0],graph_arr[1]))
print(jaccard_graph_similarity(graph_arr[0],graph_arr[2]))



# messages = [
#     {"role": "system", "content": "You are an expert in information extraction."},
#     {"role": "user", "content": real_prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)
#
# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
# print("--------")

#
# graph=triples_to_graph(new_triple)
# print(graph)
# for value in response:
#     print(value)