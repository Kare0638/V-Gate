import sys
import os
import time 


# 这是一个小技巧：确保脚本能找到上一级的 vgate 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgate.engine import VGateEngine

def run_benchmark():
    print("🚀 Initializing Engine for Benchmark...")
    engine = VGateEngine()
    
    # 测试用例：让它写一段长一点的代码，这样能测出稳定的 TPOT
    prompt = "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
    print(f"\n📝 Prompt: {prompt}")
    print("⏳ Generating... (Please wait)")
    
    # 调用我们刚升级过的 generate 方法
    result = engine.generate(prompt, max_tokens=512)
    
    # 打印结果
    print("\n" + "="*50)
    print("📊 V-Gate Performance Report (RTX 3060)")
    print("="*50)
    
    # 转换单位让看起来更直观
    ttft_ms = result['ttft'] * 1000
    tpot_ms = result['tpot'] * 1000
    tokens_per_sec = 1 / result['tpot'] if result['tpot'] > 0 else 0
    
    print(f"Generated Tokens: {result['total_tokens']}")
    print("-" * 30)
    print(f"⚡ TTFT (首字延迟):     {ttft_ms:.2f} ms")
    print(f"🔄 TPOT (生成速度):     {tpot_ms:.2f} ms/token")
    print(f"🚀 Throughput (吞吐量): {tokens_per_sec:.2f} tokens/s")
    print("-" * 30)
    
    # 简单的性能评估逻辑
    if ttft_ms < 200:
        print("✅ Latency Status: Excellent (<200ms)")
    elif ttft_ms < 500:
        print("⚠️ Latency Status: Good (<500ms)")
    else:
        print("❌ Latency Status: Slow (>500ms)")
        
    print("\nGenerated Text Preview:")
    print(result['text'][:100] + "...")

if __name__ == "__main__":
    run_benchmark()