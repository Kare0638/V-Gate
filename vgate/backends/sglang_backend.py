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

import time
from typing import Any, Dict, List

from vgate.config import ModelConfig


class SGLangBackend:
    """Inference backend using SGLang."""

    def __init__(self):
        self.engine = None

    def load_model(self, model_config: ModelConfig) -> None:
        import sglang as sgl

        print(f"Loading {model_config.model_id} with SGLang engine...")
        self.engine = sgl.Engine(
            model_path=model_config.model_id,
            mem_fraction_static=model_config.gpu_memory_utilization,
            tp_size=1,
        )

    def create_sampling_params(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> Any:
        return {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[Dict[str, Any]]:
        results = []
        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, sampling_params)
        wall_time = time.perf_counter() - start_time

        if not isinstance(outputs, list):
            outputs = [outputs]

        for output in outputs:
            text = output["text"]
            token_ids = output.get("meta_info", {}).get("completion_tokens_ids", [])
            if not token_ids:
                # Estimate token count from text length as fallback
                num_tokens = max(1, len(text.split()))
            else:
                num_tokens = len(token_ids)

            results.append({
                "text": text,
                "token_ids": list(token_ids) if token_ids else [],
                "num_tokens": num_tokens,
                "metrics": {"wall_time": wall_time / len(prompts)},
            })
        return results

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
