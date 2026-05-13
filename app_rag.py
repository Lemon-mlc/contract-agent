import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 准备数据
# 我们注释掉了下面这部分，因为我们要读自己的文件，不需要程序自动生成假数据了
# content = """
# 银行客户经理绩效考核办法：
# 第一条：存款日均增量每增加1000万，绩效加5分。
# ...
# """
# with open("bank_policy.txt", "w", encoding="utf-8") as f:
#     f.write(content)

print("📂 1. 准备读取你的文件...")

# 2. 加载文档
# 修改点：这里改成了读取你新建的 my_info.txt
loader = PyPDFLoader("test.pdf")
documents = loader.load()

# 3. 切分文本
# 修改后的切片配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # 改小！从 500 改为 300。让每个块更聚焦。
    chunk_overlap=50,    # 保持重叠，防止切断句子。
    separators=["\n\n", "\n", "。", "！", "？", "；", "，"] # 保持这个顺序，优先按段落切
)
docs = text_splitter.split_documents(documents)
print(f"✂️ 2. 文本已切分为 {len(docs)} 个片段")

# 4. 向量化并存入数据库
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()
print("💾 3. 向量数据库建立完成")

# 5. 构建提示词模板
# 修改点：角色从“银行助手”变成了“个人助手”
template = """
你是一个文档阅读助手。请根据以下上下文（Context）回答用户的问题。
如果上下文里没有答案，请直接说“我不知道”。

上下文：
{context}

问题：
{question}

回答：
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. 定义模型
llm = OllamaLLM(model="qwen2.5:3b")

# 7. 组装 RAG 链
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. 测试提问
print("\n--- 🤖 聊天模式启动 (输入 'exit' 或 'quit' 退出) ---")

while True:
    # 1. 获取用户输入
    question = input("\n👤 请输入你的问题：")

    # 2. 检查是否需要退出
    if question.lower() in ['exit', 'quit', '退出']:
        print("👋 再见！")
        break

    # 3. 调用 AI 并打印回答
    print("🤖 正在思考...", end="\r")  # 显示一个正在思考的提示
    response = rag_chain.invoke(question)
    print("🤖 答：", response)