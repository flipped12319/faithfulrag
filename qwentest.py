from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)

# prepare the model input
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
        An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total distance of 2.0 m vertically. Which of the following best describes the objects after they have fallen a distance of 1.0 m?

        Answer:
            1. your first explanation
            2. your second explanation
            3. continue as needed
        """
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)