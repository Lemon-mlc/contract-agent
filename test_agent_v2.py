from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# 工具函数
def local_weather_mock(city):
    if "北京" in city:
        return "北京今天晴转多云，气温15°C-25°C"
    elif "上海" in city:
        return "上海今天有小雨，气温18°C-22°C"
    else:
        return f"{city}的天气：晴天，20度。"

def calculator(expression):
    try:
        return str(eval(expression))
    except:
        return "表达式错误"

tools = [
    Tool(name="WeatherChecker", func=local_weather_mock, description="查询天气，输入城市名称"),
    Tool(name="Calculator", func=calculator, description="数学计算，输入表达式如 '3+5*2'"),
]

llm = Ollama(model="qwen2.5:3b", temperature=0)

# ★ 使用全英文的 ReAct 格式
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # 如果仍然超时，可以增大到 10
)

if __name__ == "__main__":
    print("测试 Agent...\n")
    resp = agent_executor.invoke({"input": "北京天气怎么样？"})
    print(f"回答：{resp['output']}\n")
    resp = agent_executor.invoke({"input": "123*456等于多少？"})
    print(f"回答：{resp['output']}\n")