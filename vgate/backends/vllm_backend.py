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

from typing import Any, Dict, List

from vgate.config import ModelConfig


class VLLMBackend:
    """Inference backend using vLLM."""

    def __init__(self):
        self.llm = None

    def load_model(self, model_config: ModelConfig) -> None:
        from vllm import LLM

        print(f"Loading {model_config.model_id} with {model_config.quantization} quantization (vLLM)...")
        self.llm = LLM(
            model=model_config.model_id,
            quantization=model_config.quantization,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            max_model_len=model_config.max_model_len,
            enforce_eager=model_config.enforce_eager,
            trust_remote_code=model_config.trust_remote_code,
        )

    def create_sampling_params(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[Dict[str, Any]]:
        outputs = self.llm.generate(prompts, sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            num_tokens = len(token_ids)

            metrics_dict = {}
            metrics = output.metrics
            if metrics:
                metrics_dict["ttft"] = metrics.first_token_time - metrics.arrival_time
                metrics_dict["gen_time"] = metrics.finished_time - metrics.first_token_time

            results.append({
                "text": text,
                "token_ids": list(token_ids),
                "num_tokens": num_tokens,
                "metrics": metrics_dict,
            })
        return results

    def shutdown(self) -> None:
        self.llm = None
