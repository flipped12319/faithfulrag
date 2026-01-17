from qwen_instruct import LLMWithThinking
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
llm_model_name="Qwen/Qwen2-7B-Instruct"
# model_name="Qwen/Qwen3-8B-FP8"
device = "cuda"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
llm=LLMWithThinking(llm_model,llm_tokenizer)
result=llm.generate("what is an apple")
print(result)