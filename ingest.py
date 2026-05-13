import os
# 【关键修改】加入国内镜像源，防止下载模型时报错
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载数据
# 假设你的文本文件叫 data.txt，放在同级目录下
loader = TextLoader('data.txt', encoding='utf-8')
documents = loader.load()

# 2. 切分文本
# 这一步很关键，不能把整本书塞给AI，要切成小块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每块500个字符
    chunk_overlap=50 # 块与块之间重叠50个字符，防止语义截断
)
texts = text_splitter.split_documents(documents)

print(f"👉 正在将文本切分为 {len(texts)} 个片段...")

# 3. 初始化 Embedding 模型
# 这里我们使用开源的 Sentence Transformers 模型
# 第一次运行会自动下载模型，大概几百兆，需要一点时间
# 注意：如果是中文内容，建议换成 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. 存入 ChromaDB
# persist_directory 指定了数据库保存在本地哪个文件夹
db = Chroma.from_documents(
    texts, 
    embeddings, 
    persist_directory="./chroma_db"
)

print("✅ 向量数据库构建完成！数据已保存在 ./chroma_db 文件夹中。")