# main.py
# -*- coding: utf-8 -*-

"""
2026年 RAG 项目适配版
针对 LangChain 最新版本 (v0.3+) 进行了路径修正和错误处理优化
"""

# 1. 导入需要的库 (注意：新版本中 TextSplitter 在 langchain_core 里)
from langchain_community.document_loaders import TextLoader
# 尝试从 community 包导入，兼容老版本
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

def main():
    print("🚀 正在启动 RAG 系统 (2026 适配版)...")
    print("-" * 50)

    # --- 第一步：加载数据 ---
    file_path = "data.txt"
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}，请确保文件存在。")
        return

    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        print(f"✅ 成功加载文档，共 {len(docs)} 页。")
    except Exception as e:
        print(f"❌ 文档加载失败: {e}")
        return

    # --- 第二步：切分文本 ---
    # 新版本推荐使用 langchain_core.text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    print(f"✂️  文本已切分为 {len(splits)} 个片段。")

    # --- 第三步：向量化并存库 ---
    print("⏳ 正在生成向量索引 (这可能需要几十秒)...")
    
    try:
        # 确保你已经运行了 ollama pull nomic-embed-text
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 【重要】新版本 Chroma 推荐使用 from_documents 的静态方法
        # 如果报错，可以尝试添加 persist_directory 参数来指定保存路径
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            # persist_directory="./chroma_db" # 可选：将数据库持久化到磁盘
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 限制返回2个片段
        print("✅ 向量库建立完成！")
        
    except Exception as e:
        print(f"❌ 向量库构建失败，请检查 Ollama 服务是否开启: {e}")
        print("💡 提示：请在终端运行 'ollama run nomic-embed-text' 或 'ollama pull nomic-embed-text'")
        return

    # --- 第四步：准备大模型 ---
    try:
        llm = Ollama(model="qwen2.5:3b", temperature=0)
        # 简单测试连接
        print("🧠 大模型连接正常。")
    except Exception as e:
        print(f"❌ 大模型连接失败: {e}")
        print("💡 请确保已运行 'ollama run qwen2.5:7b'")
        return

    # --- 第五步：构建提示词模板 ---
    template = """基于以下已知信息回答用户问题。
    如果无法从信息中找到答案，请回答 "根据已知信息无法回答该问题"。

    已知信息：
    {context}

    问题：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # --- 第六步：构建链路 ---
    # 新版本 Runnable 语法保持不变
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 第七步：开始问答 ---
    print("-" * 50)
    print("🎉 系统启动成功！输入问题开始对话 (输入 'exit' 退出)：")

    while True:
        try:
            query = input("\n❓ 我: ").strip()
            if query.lower() == "exit":
                break
            if not query:
                continue
                
            print("🤖 AI: ", end="", flush=True)
            
            # 流式输出
            for chunk in rag_chain.stream(query):
                print(chunk, end="", flush=True)
            print() # 换行
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 处理输入时出错: {e}")

    print("\n👋 再见！")

if __name__ == "__main__":
    main()