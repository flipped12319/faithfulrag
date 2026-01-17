SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS["normal"] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
SYSTEM_PROMPTS[
    "qa"] = "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."
SYSTEM_PROMPTS[
    "extract"] = "You are a precise and reliable information extractor. Your sole task is to extract relevant information from the given context strictly according to the instructions. You must not add, modify, or infer any information that is not explicitly stated in the context."
SYSTEM_PROMPTS[
    "qa-cot"] = "You are an expert in retrieval QA and Chain of Thought reasoning. Provide your reasoning steps followed by a precise and direct answer.Avoiding any unnecessary explanations or verbosity."


class PromptGenerator:
    def __init__(self, task: str = "normal", tokenizer=None):
        self.tokenizer = tokenizer
        if task == "qa":
            self.system_prompt = SYSTEM_PROMPTS["qa"]
        elif task == "extract":
            self.system_prompt = SYSTEM_PROMPTS["extract"]
        elif task == "facts":
            self.system_prompt = SYSTEM_PROMPTS["facts"]
        elif task == "qa-cot":
            self.system_prompt = SYSTEM_PROMPTS["qa-cot"]
        else:
            self.system_prompt = SYSTEM_PROMPTS["normal"]

    def _generate_prompt(self, user_prompt):
        return user_prompt

    def clean_text_prompt(self, context):
        prompt = """
        You are a text-cleaning assistant. 
        Your task is to fix **incorrect internal spaces inside words**, while keeping all correct word boundaries unchanged.

        ### Rules
        - Only remove spaces *inside a word that should be continuous*.
        - Do NOT change valid spaces between words.
        - Do NOT rewrite, paraphrase, or change punctuation.
        - Do NOT add or remove any words.
        - ONLY fix spacing errors caused by text artifacts such as:
          - "m eteor ite" → "meteorite"
          - "Observ ing" → "Observing"
          - "celestial m echanics" → "celestial mechanics" (fix only the wrong one)
          - "post -m eteor ite" → "post-meteorite"
        - Keep everything else exactly the same.
        -Only remove an internal space if the merged word is a common valid English word.If unsure, do nothing.


        ### Input text:
        {text}

        ### Output (cleaned text):
        """
        return self._generate_prompt(prompt.format(text=context))

    def generate_factual_knowledge(self, user_query):
        prompt = """
        Task Description:
        You are an expert in problem analysis. When a user presents a question, your task is to identify the factual knowledge required to answer the question. 
        Please list the relevant facts in a clear and structured manner.

        Instructions:
            Analyze the question carefully.
            Identify key areas of knowledge that are crucial for answering the question.
            Provide a brief explanation of why each area is necessary and follow the Example:

        Question:
        Who invented the theory of general relativity?

        Answer:
        To answer this question, the following areas of knowledge are required:
            1. The theory of general relativity describes gravity as a curvature of spacetime caused by mass and energy.
            2. The theory was developed by Albert Einstein in 1915.
            3. Albert Einstein is a German-born theoretical physicist who also developed the equation E = mc², which expresses the equivalence of energy and mass.

        Now, please analyze the following question:

        Question:
        {question}

        Answer:
            1. your first explanation
            2. your second explanation
            3. continue as needed
        """
        return self._generate_prompt(prompt.format(question=user_query))

    # def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
    #     prompt = """
    #     Given the following question and a set of factual knowledge, generate a background document from Wikipedia that can answer the given question. Keep the length of the document around 100 words.
    #     Question: {question}
    #     Factual Knowledge: {factual_knowledge}
    #     Background Document:
    #     """
    #     return self._generate_prompt(prompt.format(question=user_query, factual_knowledge=factual_knowledge))

    def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
        prompt = """
        You are writing a Wikipedia-style background document.

        Task:
        Using the question and the provided factual knowledge, write a coherent background document that helps answer the question.

        Rules:
        - You MUST generate a background document.
        - If the factual knowledge is incomplete, unclear, or partially irrelevant, you MUST still generate a document using your own general knowledge.
        - Do NOT explain your reasoning.
        - Do NOT mention the factual knowledge explicitly.
        - The output must be a single continuous paragraph.
        - The length must be between 80 and 120 words.
        - The output must NOT be empty.

        Question:
        {question}

        Factual Knowledge:
        {factual_knowledge}

        Background Document:
        """
        return self._generate_prompt(prompt.format(question=user_query, factual_knowledge=factual_knowledge))

    def generate_context_directly_prompt(self, user_query):
        prompt = """
           Generate a background document from Wikipedia to answer the given question:
           {question}. Keep the length of the document around 100 words
           """
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_context_extract(self, user_context):
        prompt = """
           Task Description:  
           Extract factual statements based on the given context. Each statement must be concise, accurate, and fully faithful to the information provided in the context. Avoid interpretations, opinions, or assumptions.
           Example:  

           Context:  
           The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the structure. The tower was completed in 1889 and served as the entrance arch for the 1889 World's Fair.  

           Answer:
           The following factual Statements:  
           1. The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.  
           2. The Eiffel Tower was named after the engineer Gustave Eiffel.  
           3. Gustave Eiffel's company designed and built the Eiffel Tower.  
           4. The Eiffel Tower was completed in 1889.  
           5. The Eiffel Tower served as the entrance arch for the 1889 World's Fair.

           Now, please extract the following context:
           Context:  
           {context} 

           Answer:
           1. Your first factual statement
           2. Your second factual statement  
           3. Continue as needed
           """
        return self._generate_prompt(prompt.format(context=user_context))

    def generate_qa_prompt_without_fact(self, context, question, options):
        choice_wo_facts_prompt = """
         ## Important Rules:
        - You must **think silently** — never output your internal reasoning or thoughts.
        - Use escaped quotes like \" inside strings instead of directly using "" or '' inside strings.
        - You must choose your answer from the given options

        Question:
        Which element has the highest electronegativity?

        Context:
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element. The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        CoT-Answer:
        {{
        "Reason": "According to the Context, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity."，
        "Answer": "Fluorine"
        }}
        New Example:

        Question:
        {question}

        Context:
        {context}

        Options:
        {options}

        CoT-Answer:


       """
        normal_wo_facts_prompt = """
        Task Description:
        Given a question and a context, your task is answering the question based on the context.

        Follow the steps:
        1. Analyze the **Question** carefully.
        2. Use the **Context** to provide an answer to the question.
        3.Finally, output **only** in the following strict JSON format:
            {{
                "Reason": "<short reasoning>",
                "Answer": "<final answer>"
            }}

        ## Important Rules:
        - You must **think silently** — never output your internal reasoning or thoughts.
        - Use escaped quotes like \" inside strings instead of directly using "" inside strings.
        - Output **ONLY** a valid JSON object — no markdown, no text before or after, no `<think>` tags, no explanations.
        - The JSON **must exactly match** this format:
            {{
              "Reason": "<short reasoning, one sentence only>"，
              "Answer": "<final answer, must be one of the given options>"
            }}

        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Please return in JSON format.Use escaped quotes like \" inside strings.

        Answer:
        {{
        "Reason": "According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity."，
        "Answer": "Fluorine"
        }}
        New Example:

        Question:
        {question}

        Context:
        {context}

        Answer:
        """
        if options is None:
            return normal_wo_facts_prompt.format(question=question, context=context)
        else:
            return choice_wo_facts_prompt.format(question=question, context=context, options=options)

    # def generate_qa_prompt_without_fact(self, context, question, options=None):
    #     prompt = """
    #     You are a **JSON output generator**.
    #
    #     Task Description:
    #     Given facts, a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the context.
    #
    #     Follow the steps:
    #     1. Analyze the **Question** carefully.
    #     2. Use the **Context** to provide a clear and accurate answer to the question.
    #     3. Finally, output **only** in the following strict JSON format:
    #         {{
    #             "Reason": "<short reasoning>",
    #             "Answer": "<final answer>"
    #         }}
    #
    #     ## Important Rules:
    #     - You must **think silently** — never output your internal reasoning or thoughts.
    #     - Output **ONLY** a valid JSON object — no markdown, no text before or after, no `<think>` tags, no explanations.
    #     - The JSON **must exactly match** this format:
    #         {{
    #           "Reason": "<short reasoning, one sentence only>",
    #           "Answer": "<final answer, must be one of the given options>"
    #         }}
    #
    #     Example:
    #
    #     Question:
    #     Which element has the highest electronegativity?
    #
    #     Context:
    #     Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element. The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.
    #
    #     Answer:
    #     {{
    #         "Reason": "According to the Context, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.",
    #         "Answer": "Fluorine"
    #     }}
    #
    #     ---
    #
    #     ## Your turn:
    #
    #     Question:
    #     {question}
    #
    #     Context:
    #     {context}
    #
    #     Now output the final JSON only:
    #
    #
    #     """
    #     return prompt.format(question=question, context=context)
    def generate_qa_prompt_normal_cot(self, context, question, options=None, facts=None):
        normal_w_facts_prompt = """
           You are a reasoning assistant that must output strictly valid JSON.

           ##Task Description: 
           Given facts,a question and a context, your task is to select the most accurate and relevant answer. You should only choose the answer that directly answers the question based on the facts and context.

           Follow the steps:
           1. Analyze the **Question** carefully.
           2. Use the **Facts** to provide a clear and accurate answer to the question.
           3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
           4.Finally, output **only** in the following strict JSON format:
               {{
                   "Reason": "<short reasoning>",
                   "Answer": "<final answer>"
               }}

             ## Important Rules:
               - You must **think silently** — never output your internal reasoning or thoughts.
               - Output **ONLY** a valid JSON object — no markdown, no text before or after, no `<think>` tags, no explanations.
               - Use escaped quotes like \" inside strings.
               - The JSON **must exactly match** this format:
               {{
                 "Reason": "<short reasoning, one sentence only>",
                 "Answer": "<final answer, must be one of the given options>"
               }}

           Example:

           Question:  
           Which element has the highest electronegativity?  

           Facts:  
           Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

           Context:  
           The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

           Answer:
           {{
           "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
           "Answer": Fluorine
           }}
           New Example:

           Question:
           {question}

           Facts:
           {facts}

           Context:
           {context}

           Now please return your answer to the question according to the steps
           Answer:


           """

        choice_w_facts_prompt = """
           You are a **JSON output generator**. 

           Task Description: 
           Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

           Follow the steps:
           1. Analyze the **Question** and the **Options**.
           2. Use the **Facts** to select the most accurate answer from the **Options**.
           3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
           4. Finally, generate the output

           ## Important Rules:
               - You must **think silently** — never output your internal reasoning or thoughts.
               - Output **ONLY** a valid JSON object — no markdown, no text before or after, no `<think>` tags, no explanations
               - Please answer with the **option content only**, not the letter.For example, answer like this "George Washington"
               - Use escaped quotes like \" inside strings.
               - The JSON **must exactly match** this format:
               {{
                 "Reason": "<short reasoning, one sentence only>",
                 "Answer": "<final answer, must be one of the given options>"
               }}
           Example:

           Question:  
           Which element has the highest electronegativity?  

           Facts:  
           Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

           Context:  
           The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

           Options:  
           Oxygen  
           Chlorine  
           Fluorine 

           Final Output:
           {{
           "Reason": "According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity."
           "Answer": "Fluorine"
           }}
           ---

           ## Your turn:

           Question:
           {question}

           Facts:
           {facts}

           Context:
           {context}

           Options:
           {options}

           Now output the final JSON only:
           """

        normal_wo_facts_prompt = """
           Question:  
           Which element has the highest electronegativity?  

           Context:  
           The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

           Please return in JSON format.

           CoT-Answer:
           {{
           "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
           "Answer": Fluorine
           }}
           New Example:

           Question:
           {question}

           Context:
           {context}

           CoT-Answer:
           """

        choice_wo_facts_prompt = """
           Question:  
           Which element has the highest electronegativity?  

           Context:  
           The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

           Options:  
           Oxygen  
           Chlorine  
           Fluorine 

           CoT-Answer:
           {{
           "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.“
           "Answer": Fluorine
           }}
           New Example:

           Question:
           {question}

           Context:
           {context}

           Options:
           {options}

           CoT-Answer:
           """
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, facts=facts, context=context)
        else:
            if facts is None:
                return choice_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choice_w_facts_prompt.format(question=question, facts=facts, context=context, options=options)

    def generate_qa_prompt_normal_cot_with_conflict(self, context, question, conflict_information, options=None,
                                                    facts=None):
        normal_w_facts_prompt = """
        You are a reasoning assistant that must output strictly valid JSON.

        ##Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer. You should try to choose the answer that directly answers the question based on the facts and context and conflict_information.
        The Facts and Context may contain incorrect or misleading information.All detected conflicts are explicitly provided in the **Conflict information**.

        Your goal is to answer the **Question** based on the resolved factual truth after applying conflict resolution rules.

        ## Conflict Resolution Rules (MANDATORY)
        - Each item in Conflict information corresponds to a potentially incorrect claim.
        - If "has_conflict" = "yes":
          - Discard the conflicting claim from Facts or Context.
          - Replace it with the provided "internal_knowledge".
          - The "internal_knowledge" must be treated as the ONLY valid ground truth for that claim.


        Conflict detection itself is NOT the final decision.
        Your final answer must be based on the resolved facts after applying these rules.

        ## Decision Rule (CRITICAL)
        - Always answer the Question itself, NOT whether a conflicting claim is correct or incorrect.
        - For comparison questions:
          - Explicitly compare the resolved factual values.
        - For Yes/No questions:
          - Answer "Yes" ONLY if the condition stated in the Question is satisfied by the resolved facts.
          - Answer "No" ONLY if the condition stated in the Question is NOT satisfied.
          - Do NOT answer "No" merely because a conflicting claim was false.

        Follow the steps:
        1. Analyze the **Question** carefully.
        2. Use the **Facts** and **Conflict information** to answer the question.
           - If Conflict information indicates "has_conflict" = yes:
             - Ignore or discount the corresponding conflicting facts.
             - Use the provided "internal_knowledge" field as the ground truth instead.
        3. Refer to the **Context** and **Conflict information** If the facts do not contain enough information to answer the question, or if additional information is needed.
           - If Conflict information indicates "has_conflict" = yes:
             - Ignore or discount the corresponding content of context.
             - Use the provided "internal_knowledge" field as the ground truth instead.
        4. Finally, output **only** in the following strict JSON format:
            {{
                "Reason": "<short reasoning>",
                "Answer": "<final answer>"
            }}

          ## Important Rules:
            - You must **think silently** — never output your internal reasoning or thoughts.
            - Output **ONLY** a valid JSON object — no markdown, no text before or after, no `<think>` tags, no explanations.
            - Use escaped quotes like \" inside strings.\
            - The final answer should directly answers the question

        Example(no conflicts):

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Conflict information:
        [
        ...
        {{
        "claim":"Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.",
        "has_conflict": "no",
        "Reason": "This statement aligns with the established understanding of electronegativity trends in the periodic table. Electronegativity does indeed increase from left to right across a period due to the increasing nuclear charge which pulls electrons closer to the nucleus, and it decreases from top to bottom within a group as atomic size increases, leading to less effective electron attraction.",
        "internal_knowledge": "Based on the principles of chemistry and the periodic table, the trends of electronegativity follow these rules, and Fluorine's high electronegativity is consistent with its position as the most electronegative element."
        }}
        ...
        ]

        Answer:
        {{
        "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
        "Answer": Fluorine
        }}

        Example(with conflicts):

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Oxygen is the most electronegative element. 

        Context:  
        The Pauling scale is commonly used to measure electronegativity. Oxygen and fluorine are both highly electronegative nonmetals, and oxygen plays a crucial role in oxidation reactions.

        Conflict information:
        [
        ...
          {{
            "claim": "Electronegativity increases across periods and decreases down groups. Oxygen is the most electronegative element.",
            "has_conflict": "yes",
            "Reason": "While the general trend of electronegativity increasing across periods and decreasing down groups is correct, the claim that oxygen is the most electronegative element is incorrect. According to established chemical knowledge and the Pauling scale, fluorine has a higher electronegativity than oxygen.",
            "internal_knowledge": "Based on standard chemistry references and the Pauling electronegativity scale, fluorine has the highest electronegativity of all elements. Oxygen is highly electronegative but ranks below fluorine."
          }}
        ...
        ]


        Answer:
        {{
          "Reason": "Although the provided facts claim that oxygen is the most electronegative element, the conflict information indicates a contradiction with established chemical knowledge. According to the model's internal knowledge and the Pauling scale, fluorine has a higher electronegativity than oxygen. Therefore, despite the conflicting fact, the answer is derived by prioritizing the more reliable internal knowledge.",
          "Answer": "Fluorine"
        }}

        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Conflict information:
        {conflict_information}

        Now please return your answer to the question according to the steps
        Answer:


        """

        choice_w_facts_prompt = """
        You are a **JSON output generator**. 

        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** and the **Options**.
        2. Use the **Facts** to select the most accurate answer from the **Options**.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Finally, generate the output

        ## Important Rules:
            - You must **think silently** — never output your internal reasoning or thoughts.
            - Output **ONLY** a valid JSON object — no markdown, no text before or after, no `<think>` tags, no explanations
            - Please answer with the **option content only**, not the letter.For example, answer like this "George Washington"
            - Use escaped quotes like \" inside strings.
            - The JSON **must exactly match** this format:
            {{
              "Reason": "<short reasoning, one sentence only>",
              "Answer": "<final answer, must be one of the given options>"
            }}
        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        Final Output:
        {{
        "Reason": "According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity."
        "Answer": "Fluorine"
        }}
        ---

        ## Your turn:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Options:
        {options}

        Now output the final JSON only:
        """

        normal_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Please return in JSON format.

        CoT-Answer:
        {{
        "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
        "Answer": Fluorine
        }}
        New Example:

        Question:
        {question}

        Context:
        {context}

        CoT-Answer:
        """

        choice_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        CoT-Answer:
        {{
        "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.“
        "Answer": Fluorine
        }}
        New Example:

        Question:
        {question}

        Context:
        {context}

        Options:
        {options}

        CoT-Answer:
        """
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, facts=facts, context=context,
                                                    conflict_information=conflict_information)
        else:
            if facts is None:
                return choice_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choice_w_facts_prompt.format(question=question, facts=facts, context=context, options=options)

    def generate_qa_prompt_schedule_cot(self, context, question, facts, options=None, task='multiple-choice',
                                        example=True):
        normal_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** carefully.
        2. Use the **Facts** to provide a clear and accurate answer to the question.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Please return in JSON format: Reason: (reason) Answer:(answer)


        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Answer:
        {{
        "Reason":  
        1. [Fact Analysis]: Facts explicitly state "Fluorine is the most electronegative element"  
        2. [Initial Judgment]: Based on the explicit statement in the facts, fluorine is identified as the element with the highest electronegativity.  
        3. [Context Check]: The context provides additional support by mentioning the Pauling scale and comparing fluorine with chlorine, reinforcing that fluorine has the highest electronegativity. No further information is needed. 
        4. [Final Verification]: The facts and context are consistent and conclusive, confirming that fluorine is the correct answer. 
        "Answer": Fluorine
        }}

        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Now please give your answer according to the given information and steps;
        Answer:
        """
        choice_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** and the **Options**.
        2. Use the **Facts** to select the most accurate answer from the **Options**.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Please return in JSON format: Reason: (reason) Answer:(answer)

        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        Please return in JSON format.
        CoT-Answer:
        {{
            "Reason":  
            1. [Fact Analysis]: Facts explicitly state "Fluorine is the most electronegative element"  
            2. [Option Matching]: Option C directly matches the factual declaration  
            3. [Context Check]: No contextual supplementation needed - Facts provide conclusive evidence  
            4. [Final Verification]: No conflicting information; perfect match with Option C  
            "Answer": Fluorine
        }}

        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Options:
        {options}

        CoT-Answer:
        """

        normal_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Please return in JSON format.

        CoT-Answer:
        {{
            "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
            "Answer": Fluorine
        }}

        New Example:

        Question:
        {question}

        Context:
        {context}

        CoT-Answer:
        """

        choice_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        CoT-Answer:
        {{
        "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
        "Answer": Fluorine
        }}
        New Example:

        Question:
        {question}

        Context:
        {context}

        Options:
        {options}

        CoT-Answer:
        """
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, facts=facts, context=context)
        else:
            if facts is None:
                return choice_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choice_w_facts_prompt.format(question=question, facts=facts, context=context, options=options)

    def generate_qa_prompt(self, context, question, options=None, facts=None):
        normal_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the answer that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** carefully.
        2. Use the **Facts** to provide a clear and accurate answer to the question.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.

        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Answer: Fluorine
        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Answer:
        """
        choices_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** and the **Options**.
        2. Use the **Facts** to select the most accurate answer from the **Options**.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Please directly answer the option you want to choose. No modification is allowed.

        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        Answer: Fluorine

        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Options:
        {options}

        Answer:
        """
        normal_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Answer: Fluorine

        New Example:

        Question:
        {question}

        Context:
        {context}

        Answer:
        """
        choices_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:
        Oxygen
        Chlorine
        Fluorine

        Answer: Fluorine

        New Example:

        Question:
        {question}

        Context:
        {context}

        Options:
        {options}

        Answer:
        """
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, context=context, facts=facts)
        else:
            if facts is None:
                return choices_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choices_w_facts_prompt.format(question=question, context=context, options=options, facts=facts)

    def extract_tripple(self, context):
        prompt = """
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
        return self._generate_prompt(prompt.format(text=context))

    def generate_conflict_prompt(self, conflictType, context):
        prompt = """
        **IMPORTANT**:
        The output will be parsed by a strict JSON parser.
        If the output contains ```json, ``` or any markdown fences,the output will be considered INVALID and discarded.


        You are a **JSON output generator** for conflict detection in RAG systems.. 

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
        7. Return a raw JSON object ONLY.
        Do NOT include ```json, ```, markdown, comments, or explanations.
        Any output containing ``` will be treated as a failure.
        8.All quotation marks inside strings MUST be escaped as \".
        Do not use unescaped double quotes inside values.

        =====================================================================
        ATTRIBUTE-TYPE SPECIAL RULES  
        =====================================================================
        If conflict_type = "attribute", you MUST:

        A. Identify at least 2–5 candidate “attribute properties” from the context  
           (e.g., nationality, occupation, birthplace, species, color, size, language, organization, role).  
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

        VALID OUTPUT EXAMPLE:
        {{...}}

        INVALID OUTPUT EXAMPLE:
        ```json
        {{...}}

        type:
        {type}

        context:
        {your_input_fact}
        """
        return self._generate_prompt(prompt.format(type=conflictType, your_input_fact=context))

    def information_extraction(self, chunk):
        prompt = """
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

        return self._generate_prompt(prompt.format(chunk=chunk))

    def conflict_detection_layer1(self, chunk):
        prompt = """
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

        return self._generate_prompt(prompt.format(chunk=chunk))

    def conflict_detection_layer2(self, claim):
        prompt = """
        You are a factual verifier.

        Given a single factual statement, determine whether it conflicts with well-established world knowledge
        based on your own internal knowledge (not external sources).

        Statement:
        "{claim}"

        Output:{{
            "claim":"{claim}",
            "has_conflict":"...",
            "Reason":"...",
            "internal_knowledge":"..."
        }}
        Rules:
        Output JSON only
        - Use double quotes (") for all strings and keys.
        - Do NOT use single quotes.
        About has_conflict in Output
        - Use "yes" if the statement contradicts well-established world knowledge.
        - Use "no" if the statement does not contradict well-established world knowledge.
        - Use "uncertain" only if you are not confident about the conflict or you don't have the internal_knowledge about the claim.
        """

        return self._generate_prompt(prompt.format(claim=claim))