import os
from vllm import LLM, SamplingParams

class VGateEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading {model_name} with AWQ quantization for RTX 3060 optimization...")
        self.llm = LLM(
            model=model_name,
            quantization="awq",           # 核心：开启量化，解决 Issue #1
            gpu_memory_utilization=0.7,   # 预留 30% 显存
            max_model_len=2048,           # 限制上下文
            enforce_eager=True,           # 关闭 CUDA Graph
            trust_remote_code=True
        )

    def generate(self, prompt, max_tokens=256):
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text