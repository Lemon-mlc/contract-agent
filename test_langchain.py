# test_langchain.py

from langchain_ollama import OllamaLLM

print("🚀 正在尝试连接本地模型...")

# 初始化模型，确保这里的名字和你 Ollama 里的一致
llm = OllamaLLM(model="qwen2.5:3b")

print("✅ 模型加载成功，正在提问...")

# 发送一个简单的测试问题
response = llm.invoke("请用一句话介绍你自己")

print("🤖 模型回答：")
print(response)