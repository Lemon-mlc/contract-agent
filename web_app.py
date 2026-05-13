import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="智能合同助手", page_icon="📄")
st.title("📄 智能合同问答助手")
st.markdown("上传您的合同文件，我将基于文档内容回答您的问题。")
# --- Prompt 模板 ---

template = """
你是一名专业合同分析助手。

请严格依据合同内容回答问题。

规则：
1. 不允许编造
2. 如果合同中未提及，直接回答“合同中未提及”
3. 回答尽量准确专业
4. 涉及金额、日期、责任时要明确指出

合同内容：
{context}

问题：
{question}

回答：
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# --- 2. 初始化/加载模型 (使用缓存加速，避免重复加载) ---
@st.cache_resource
def load_models():
    # 这里使用和你之前一样的模型配置
    embeddings = OllamaEmbeddings(model="nomic-embed-text") # 或者你用的 mxbai-embed-large
    llm = Ollama(model="qwen2.5:3b", temperature=0)
    return embeddings, llm

# --- 3. 处理文件并建立索引 ---
def process_document(file):
    embeddings, _ = load_models()
    
    # 保存上传的临时文件
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
        
    # 1. 加载 PDF
    loader = PyMuPDFLoader("temp.pdf")
    pages = loader.load()
    
    # 2. 切片 (这里应用了我们之前优化的参数)
    # 尝试使用更小的块，并优先按段落和句子分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；"
        ]
    )
    docs = text_splitter.split_documents(pages)
    import hashlib

    file_hash = hashlib.md5(
        file.name.encode("utf-8")
    ).hexdigest()

    index_path = f"faiss_{file_hash}"
    
    # 3. 向量化并存入数据库
    with st.spinner("正在学习文档内容，请稍候..."):

        if os.path.exists(index_path):

            db = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:

            db = FAISS.from_documents(docs, embeddings)

            db.save_local(index_path)
    
    # 删除临时文件
    os.remove("temp.pdf")
    
    return db

# --- 4. 主界面逻辑 ---
# 文件上传组件
uploaded_file = st.file_uploader("上传 PDF 合同文件", type="pdf")
# 检测是否上传了新文件
if uploaded_file is not None:

    if (
        "current_file" not in st.session_state
        or st.session_state.current_file != uploaded_file.name
    ):

        # 新文件时清空旧状态
        st.session_state.current_file = uploaded_file.name

        if "db" in st.session_state:
            del st.session_state.db

        if "qa_chain" in st.session_state:
            del st.session_state.qa_chain

        if "messages" in st.session_state:
            st.session_state.messages = []

# 如果上传了文件
if uploaded_file is not None:
    # 只有在 session_state 中没有数据库时才处理（避免重复处理）
    if "db" not in st.session_state:
        st.session_state.db = process_document(uploaded_file)
        st.success("文档学习完成！现在可以提问了。")
    
    if "qa_chain" not in st.session_state:
        _, llm = load_models()

        retriever = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10
            }
        )

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT
            }
        )

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入框
    if prompt := st.chat_input("请输入关于合同的问题..."):
        # 1. 显示用户的问题
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                # 调用模型
                response = st.session_state.qa_chain({
                    "query": prompt
                })
                answer = response["result"]
                # 显示回答
                st.markdown(answer)
                # 显示参考内容
                with st.expander("📌 查看合同依据"):

                    for i, doc in enumerate(response["source_documents"]):

                        st.markdown(f"### 参考片段 {i+1}")

                        st.write(doc.page_content)

                        st.divider()
                
                # 保存聊天记录
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

else:
    st.info("请先上传 PDF 文件以开始对话。")