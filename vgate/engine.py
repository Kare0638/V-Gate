import os
import time
from vllm import LLM, SamplingParams

class VGateEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct-AWQ"):
        print(f"Loading {model_name} with AWQ quantization for RTX 3060 optimization...")
        self.llm = LLM(
            model=model_name,
            quantization="awq",           # 核心: 开启量化
            gpu_memory_utilization=0.7,   # 预留 30% 显存给 KV Cache
            max_model_len=2048,           # 限制上下文
            enforce_eager=True,           # 关闭 CUDA Graph 以节省显存
            trust_remote_code=True
        )

    def chat_completions(self, prompt, max_tokens=256):
        # 构造采样参数
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
        
        # 🟢 1. 开始手动计时 (Wall Clock Time)
        start_time = time.perf_counter()
        
        # 执行推理
        outputs = self.llm.generate([prompt], sampling_params)
        
        # 🔴 2. 结束计时
        end_time = time.perf_counter()
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        
        # 🟡 3. 获取 Metrics (带 Fallback 机制)
        metrics = output.metrics
        
        if metrics:
            # 如果 vLLM 给了内部数据，优先使用 (更准)
            ttft = metrics.first_token_time - metrics.arrival_time
            total_time = metrics.finished_time - metrics.first_token_time
        else:
            # 🛡️ Fallback: 如果 metrics 是 None，使用手动计时
            print("⚠️ Warning: vLLM internal metrics missing. Using wall-clock time.")
            ttft = 0.0  # 离线模式下很难测准 TTFT，暂置为 0
            total_time = end_time - start_time

        # 计算 TPOT (避免除以零)
        tpot = (total_time / num_tokens) if num_tokens > 0 else 0
        
        return {
            "text": generated_text,
            "ttft": ttft,
            "tpot": tpot,
            "total_tokens": num_tokens
        }

    def embeddings(self, input_text: str):
        """
        Placeholder for embeddings generation.
        In a real scenario, a dedicated embedding model would be loaded and used here.
        Returns a mock embedding for MVP.
        """
        print(f"VGateEngine: Generating mock embeddings for input: '{input_text}'")
        # Return a fixed mock embedding for now
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [i * 0.01 for i in range(1536)], # A longer mock embedding
                    "index": 0,
                }
            ],
            "model": "mock-embedding-model",
            "usage": {"prompt_tokens": len(input_text), "total_tokens": len(input_text)},
        }