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