import os
import traceback
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 全局对话历史存储 =====
chat_histories: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_ROUNDS = 5

# ===== LangChain 导入 =====
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# ===== 工具定义 =====
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="用于数学计算。输入应为数学表达式，如 '3+5*2'。"
)

def date_calculator(input_str: str) -> str:
    try:
        parts = input_str.split()
        date_str = parts[0]
        days = int(parts[1])
        from datetime import datetime, timedelta
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        result_dt = dt + timedelta(days=days)
        return result_dt.strftime("%Y-%m-%d")
    except Exception as e:
        return f"日期计算错误: {str(e)}"

date_tool = Tool(
    name="DateCalculator",
    func=date_calculator,
    description="用于日期加减计算。输入格式为 'YYYY-MM-DD +天数' 或 'YYYY-MM-DD -天数'，例如 '2026-05-10 +3'"
)

def create_knowledge_tool(retriever):
    def knowledge_search(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return "\n".join([doc.page_content for doc in docs])
    return Tool(
        name="KnowledgeBase",
        func=knowledge_search,
        description="用于检索公司报销流程规范、合同条款等知识。输入应为一个问题或关键词。"
    )

# ===== 初始化向量数据库和 LLM =====
logger.info("正在加载向量数据库...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})
logger.info("✅ 向量数据库加载完成")

logger.info("正在加载大模型 qwen2.5:3b...")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = Ollama(model="qwen2.5:3b", temperature=0, base_url=ollama_base_url)
logger.info("✅ 大模型加载完成")

# ===== 创建工具列表 =====
knowledge_tool = create_knowledge_tool(retriever)
tools = [knowledge_tool, calculator_tool, date_tool]

# ===== 创建 ReAct Agent（含历史上下文） =====
react_template = """You are a helpful assistant with access to the following tools:

{tools}

Conversation history:
{history}

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
    max_iterations=10,
    return_intermediate_steps=True
)
logger.info("✅ Agent 创建完成")

# ===== FastAPI 应用 =====
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="合同智能问答API (Agent 版)",
    description="基于 RAG + Agent 的合同文档问答接口，支持多轮对话",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    conversation_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str
    success: bool = True

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"收到问题: {request.question} (会话: {request.conversation_id})")

        # 获取并裁剪历史
        history = chat_histories.get(request.conversation_id, [])
        recent_history = history[-(MAX_HISTORY_ROUNDS * 2):]
        history_text = ""
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        if not history_text:
            history_text = "(无历史对话)"

        # 调用 Agent
        response = agent_executor.invoke({
            "input": request.question,
            "history": history_text
        })
        answer = response.get("output", "")

        # 提取 sources
        sources = []
        intermediate_steps = response.get("intermediate_steps", [])
        for step in intermediate_steps:
            action, observation = step
            if action.tool == "KnowledgeBase":
                obs_str = str(observation) if observation else ""
                sources.append(obs_str[:500])

        # 保存本轮对话
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": answer})
        chat_histories[request.conversation_id] = history

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=request.conversation_id,
            success=True
        )
    except Exception as e:
        error_msg = f"处理请求时出错: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="服务暂时不可用，请稍后再试")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "contract-qa-api-agent",
        "model": "qwen2.5:3b",
        "tools": [t.name for t in tools],
        "retriever_k": 5,
        "memory": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)