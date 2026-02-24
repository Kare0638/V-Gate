# Copyright 2025 the V-Gate authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Optional

from vgate.config import ModelConfig, get_config
from vgate.tracing import get_tracer

tracer = get_tracer("vgate.engine")

DRY_RUN = os.getenv("VGATE_DRY_RUN", "false").lower() in ("true", "1", "yes")


class _DryRunLLM:
    """Mock LLM that returns placeholder responses without GPU."""

    def generate(self, prompts, sampling_params):
        from unittest.mock import MagicMock
        outputs = []
        for prompt in prompts:
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock()]
            mock_output.outputs[0].text = f"[dry-run] echo: {prompt[:80]}"
            mock_output.outputs[0].token_ids = list(range(8))
            mock_output.metrics = None
            outputs.append(mock_output)
        return outputs


class VGateEngine:
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initialize the vLLM engine with configuration.

        Args:
            model_config: Model configuration. If None, uses global config.
        """
        if DRY_RUN:
            print("V-Gate starting in DRY-RUN mode (no GPU required)")
            self.llm = _DryRunLLM()
            return

        from vllm import LLM, SamplingParams  # noqa: F811

        if model_config is None:
            model_config = get_config().model

        print(f"Loading {model_config.model_id} with {model_config.quantization} quantization...")
        self.llm = LLM(
            model=model_config.model_id,
            quantization=model_config.quantization,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            max_model_len=model_config.max_model_len,
            enforce_eager=model_config.enforce_eager,
            trust_remote_code=model_config.trust_remote_code,
        )

    def chat_completions(self, prompt, max_tokens=256):
        with tracer.start_as_current_span("engine.chat_completions") as span:
            span.set_attribute("prompt_length", len(prompt))
            span.set_attribute("max_tokens", max_tokens)
            span.set_attribute("dry_run", DRY_RUN)

            # 构造采样参数
            if DRY_RUN:
                sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": max_tokens}
            else:
                from vllm import SamplingParams
                sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)

            start_time = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            end_time = time.perf_counter()

            output = outputs[0]
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)

            metrics = output.metrics

            if metrics:
                ttft = metrics.first_token_time - metrics.arrival_time
                total_time = metrics.finished_time - metrics.first_token_time
            else:
                print("Warning: vLLM internal metrics missing. Using wall-clock time.")
                ttft = 0.0
                total_time = end_time - start_time

            tpot = (total_time / num_tokens) if num_tokens > 0 else 0

            span.set_attribute("tokens_generated", num_tokens)
            span.set_attribute("ttft_ms", round(ttft * 1000, 2))

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