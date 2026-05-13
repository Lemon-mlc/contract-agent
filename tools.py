# tools.py
from langchain.tools import Tool
from datetime import datetime, timedelta

# ------------- 知识库检索工具（使用你已经初始化的 retriever） -------------
def create_knowledge_tool(retriever):
    """接收 retriever 对象，返回一个 Tool 实例"""
    def knowledge_search(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        # 将检索到的文档片段合并为一段文本
        return "\n".join([doc.page_content for doc in docs])
    
    return Tool(
        name="KnowledgeBase",
        func=knowledge_search,
        description="用于检索公司报销流程规范、合同条款等知识。输入应为一个问题或关键词。"
    )

# ------------- 计算器工具 -------------
def calculator(expression: str) -> str:
    """执行数学计算，例如 '123*456' 或 '(3+5)*2' """
    try:
        # 使用 eval 有安全风险，建议在可信环境中使用，或者用 ast.literal_eval 等
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="用于数学计算。输入应为数学表达式，如 '3+5*2'。"
)

# ------------- 日期计算工具 -------------
def date_calculator(input_str: str) -> str:
    """
    计算日期加减。输入格式：'YYYY-MM-DD +N' 或 'YYYY-MM-DD -N'
    例如: '2026-05-10 +3' 返回 '2026-05-13'
    """
    try:
        parts = input_str.split()
        date_str = parts[0]
        days = int(parts[1])
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