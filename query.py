import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 引入大模型相关库
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 2. 加载向量数据库
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2}) # 只取最相关的2段

# 3. 初始化本地大模型
# 确保你已经运行了 ollama run qwen
llm = Ollama(model="qwen2.5:3b") 

# 4. 定义提示词
# 这里我们告诉 AI：你是一个助手，请根据下面的资料回答问题
template = """
你是一个乐于助人的公司行政助手。
请使用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道，不要编造答案。

<context>
{context}
</context>

问题: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. 创建链路
# question_answer_chain：负责把资料和问题喂给大模型
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain：负责先从数据库找资料，再喂给大模型
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. 开始对话
print("🤖 RAG 系统已启动 (输入 'quit' 退出)")
while True:
    query = input("\n👦 你: ")
    if query.lower() == 'quit':
        break
    
    print("🤖 AI 正在思考...")
    # 调用链路
    response = rag_chain.invoke({"input": query})
    
    # 打印答案
    print(f"🗣️ 回答: {response['answer']}")