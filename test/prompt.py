SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS["normal"] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
SYSTEM_PROMPTS["qa"] = "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."
SYSTEM_PROMPTS["extract"] = "You are a precise and reliable information extractor. Your sole task is to extract relevant information from the given context strictly according to the instructions. You must not add, modify, or infer any information that is not explicitly stated in the context."
SYSTEM_PROMPTS["qa-cot"] = "You are an expert in retrieval QA and Chain of Thought reasoning. Provide your reasoning steps followed by a precise and direct answer.Avoiding any unnecessary explanations or verbosity."

class PromptGenerator:
    def __init__(self, task:str="normal", tokenizer=None):
        self.tokenizer = tokenizer
        if task=="qa":
            self.system_prompt = SYSTEM_PROMPTS["qa"]
        elif task=="extract":
            self.system_prompt = SYSTEM_PROMPTS["extract"]
        elif task=="facts":
            self.system_prompt = SYSTEM_PROMPTS["facts"]
        elif task=="qa-cot":
            self.system_prompt = SYSTEM_PROMPTS["qa-cot"]
        else:
            self.system_prompt = SYSTEM_PROMPTS["normal"]

    def _generate_prompt(self, user_prompt):
        return user_prompt

    def clean_text_prompt(self,context):
        prompt="""
         You are a strict text-normalization assistant.
        
        Your ONLY task:
        - Fix incorrect spaces that appear *inside a single word*.
        - Keep all valid spaces between separate words.
        - Do NOT add or remove punctuation.
        - Do NOT rewrite, rephrase, or format the text.
        - Do NOT add explanations, comments, or code blocks.
        - Do NOT add or remove line breaks.
        - If you are not certain that removing the space will form a valid English word, do nothing.
        
        Hard rule (very important):
        - If the input contains two words without a space (e.g., "rigoroustesting"), **you must NOT insert a space**.
        - If the input contains two words WITH a space (e.g., "rigorous testing"), **you must NOT remove that space**.
        
        Examples of valid fixes:
        "m eteor ite" → "meteorite"
        "Observ ing" → "Observing"
        "post -m eteor ite" → "post-meteorite"
        "rotat ion" → "rotation"
        
        Examples of things NOT allowed:
        - Do NOT join two separate words ("planet surface" must not become "planetsurface").
        - Do NOT remove or add normal word boundaries.
        - Do NOT add ``` or other markdown formatting.
        
        Now return ONLY the cleaned text, with no explanation.
        
        Input:
        {text}
        
        Output:
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


    def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
        prompt = """
        Given the following question and a set of factual knowledge, generate a background document from Wikipedia that can answer the given question. Keep the length of the document around 100 words.
        Question: {question}
        Factual Knowledge: {factual_knowledge}
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
        choice_wo_facts_prompt="""
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
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

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

        CoT-Answer:
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
        CoT-Answer:
        
     
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
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

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
