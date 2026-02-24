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

from vgate.engine import VGateEngine

def smoke_test():
    print("Initializing V-Gate Engine...")
    # 这里我们初始化引擎
    engine = VGateEngine()
    
    prompt = "Hello, tell me a verified fact about Sydney."
    print(f"Prompting: {prompt}")
    
    # 生成回复
    response = engine.chat_completions(prompt)
    
    print("-" * 50)
    print("Model Response:")
    print(response)
    print("-" * 50)
    print("✅ Engine is working!")

if __name__ == "__main__":
    smoke_test()