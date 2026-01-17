

class LLMWithThinking:
    def __init__(self, model,tokenizer,):

        self.model=model
        self.tokenizer = tokenizer
        self.device = "cuda"

    def generate(self,system_prompt,prompt_list, max_new_tokens: int = 512) -> dict:
        """
        生成模型输出，并分离思考内容（<think>）和最终回答。
        """
        messages_list=[]
        for prompt in prompt_list:
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

        print("messagelist  ",messages_list)
        # 构造输入模板（支持“思考模式”）

        batch_texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_list
        ]
        model_inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # 4. 批量生成
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )

        outputs = []
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            outputs.append(output_ids[len(input_ids):])

        # ========== 6. 批量解码 ==========
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses

    def generate_all(self, system_prompt, prompt_list, batch_size=8, max_new_tokens=512):
        """面对大量 prompt，自动分批处理"""
        all_outputs = []

        for start in range(0, len(prompt_list), batch_size):
            end = start + batch_size
            batch = prompt_list[start:end]

            print(f"Processing batch {start} → {end-1}")

            batch_outputs = self.generate(
                system_prompt,
                batch,
                max_new_tokens=max_new_tokens
            )

            all_outputs.extend(batch_outputs)

        return all_outputs

