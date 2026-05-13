import os
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# --- 1. 定义一个绝对能成功的“本地天气工具” ---
def local_weather_mock(city):
    """
    这是一个本地模拟函数，不联网，直接返回预设数据。
    用来保证程序在任何网络环境下都能跑通。
    """
    # 简单的关键词匹配
    if "北京" in city:
        return "北京今天晴转多云，气温 15°C - 25°C，微风，空气质量优。"
    elif "上海" in city:
        return "上海今天有小雨，气温 18°C - 22°C，东北风3-4级。"
    else:
        return f"{city}的天气数据暂未收录，但假装这里是晴天，气温20度。"

# --- 2. 初始化大模型 ---
# 确保你的 Ollama 服务正在运行 (ollama serve)
llm = Ollama(model="qwen2.5-coder:latest", temperature=0)

# --- 3. 定义工具 ---
weather_tool = Tool(
    name="WeatherChecker",
    func=local_weather_mock,
    description="当你需要查询天气时使用。输入应该是城市名称。"
)

tools = [weather_tool]

# --- 4. 初始化 Agent ---
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# --- 5. 运行 ---
print("--- AI 助手已就绪 (输入 'exit' 退出) ---")
while True:
    query = input("\n你: ")
    if query.lower() == 'exit':
        break

    # 注意：新版 LangChain 推荐使用 invoke，但 run 也能用
    try:
        response = agent.invoke({"input": query})
        # 如果返回的是字典，提取输出
        if isinstance(response, dict):
            print(f"AI: {response.get('output')}")
        else:
            print(f"AI: {response}")
    except Exception as e:
        print(f"AI: 发生错误: {e}")