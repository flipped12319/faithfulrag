from vllm import LLM, SamplingParams

# 指定采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# 加载模型（支持 Hugging Face 上的模型）
llm = LLM(model="Qwen/Qwen3-8B",trust_remote_code=True,max_model_len=32000)

# 输入提示词
prompt = "please explain yourself."

# 执行推理
outputs = llm.generate([prompt], sampling_params)

# 输出结果
for output in outputs:
    print(output.outputs[0].text)