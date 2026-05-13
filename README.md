# 智能合同助手 — 从零搭建到完整部署（RAG + Agent + 多轮对话）

## 项目概览
本项目是一个基于 **RAG（检索增强生成）** + **ReAct Agent** 的企业报销/合同问答系统。用户上传 PDF/TXT 文档后，AI 能根据文档内容回答专业问题，并自动调用计算器、日期计算等工具完成辅助任务。整个系统采用前后端分离架构，通过 Docker Compose 一键部署。

---

## 技术栈
| 组件 | 技术 |
|------|------|
| 语言模型 | Qwen2.5:3b（通过 Ollama 部署） |
| Embedding 模型 | sentence-transformers/all-MiniLM-L6-v2 |
| 向量数据库 | ChromaDB |
| Agent 框架 | LangChain（ReAct Agent） |
| 后端 | FastAPI + Uvicorn |
| 前端 | Streamlit |
| 容器化 | Docker + Docker Compose |
| 开发语言 | Python 3.11 |

---

## 一、环境准备

### 1.1 安装系统依赖
- **Python** 3.8~3.11（推荐 3.11）
- **Docker Desktop**（Windows） / Docker Engine（Linux）
- **Git**（可选）

### 1.2 克隆项目
```bash
git clone https://github.com/Lemon-mlc/contract-agent.git   # 你的项目地址
cd contract-agent
```

### 1.3 创建 Python 虚拟环境（用于本地调试，非必须）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

---

## 二、数据准备

### 2.1 准备知识文档
将你的公司报销流程、合同等文本保存为 `data.txt`（UTF-8 编码）。示例内容格式见项目根目录的 `data.txt`。

### 2.2 运行 ingest.py 生成向量库
确保虚拟环境已激活，且已安装必要包（见 `requirements.txt`）。
```bash
pip install -r requirements.txt
python ingest.py
```
成功后会在当前目录生成 `chroma_db` 文件夹。

**常见问题**：
- **ModuleNotFoundError: No module named 'langchain.text_splitter'** → 新版 LangChain 需改为 `from langchain_text_splitters import RecursiveCharacterTextSplitter`。
- **HuggingFace 连接超时** → 在 `ingest.py` 中添加 `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` 使用国内镜像。

---

## 三、后端开发（FastAPI + Agent）

### 3.1 核心文件
- `api.py`：FastAPI 应用，包含 RAG、Agent、多轮对话记忆。
- `tools.py`（可选，可内嵌在 `api.py` 中）：自定义工具函数。

### 3.2 启动后端（本地测试）
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 3.3 遇到的问题及解决
| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'langchain.chains'` | LangChain 版本升级后移除了 `chains` 模块 | 安装 `langchain==0.2.16` 以及对应 `langchain-community==0.2.16` |
| `ValueError: This function requires a .bind_tools method` | 使用了旧版 `Ollama` 类而不是 `ChatOllama` | 改用 `from langchain_community.chat_models import ChatOllama`，或使用 `create_react_agent`（不要求 bind_tools） |
| `NameError: name 'llm' is not defined` | 变量定义顺序错误，`llm` 在调用后才定义 | 确保 `llm = Ollama(...)` 在 `create_react_agent(...)` 之前 |
| `500 Internal Server Error` （无详细信息） | `except` 块没有打印 traceback | 添加 `logger.error(traceback.format_exc())` |
| 参考来源不显示 | `return_intermediate_steps=True` 未设置，或 `sources` 提取逻辑错误 | 在 `AgentExecutor` 中添加 `return_intermediate_steps=True`，并在 `/chat` 端点中解析 `intermediate_steps` |

---

## 四、前端开发（Streamlit）

### 4.1 核心文件
- `app.py`：Streamlit 界面，通过 HTTP 调用后端 API。

### 4.2 遇到的重要问题

**问题：前端报错 `Failed to resolve 'backend'`**
原因：前端代码中 `API_URL` 通过环境变量设置为 `http://backend:8000`，但宿主机上无法解析 Docker 内部主机名。
解决：在 `app.py` 中实现自动检测：

```python
def get_api_url():
    import socket
    try:
        socket.gethostbyname("backend")
        return "http://backend:8000"
    except socket.gaierror:
        return "http://localhost:8000"
```

同时注释掉 `docker-compose.yml` 中 frontend 的 `environment` 中的 `API_URL`，让容器也使用自动检测。

**问题：前端请求报错 `HTTPConnectionPool(host='localhost', port=8000): Connection refused`**
原因：后端容器未启动或端口映射失效。
解决：
```bash
docker-compose ps backend          # 检查状态
docker-compose logs --tail=20 backend  # 查看日志
docker-compose restart backend      # 重启
```

---

## 五、Docker 容器化

### 5.1 文件结构
```
project/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
├── api.py
├── app.py
├── ingest.py
├── data.txt
├── chroma_db/               # 由 ingest.py 生成
└── requirements.txt
```

### 5.2 Dockerfile.backend
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py .
COPY ingest.py .
COPY data.txt .
COPY chroma_db ./chroma_db
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.3 Dockerfile.frontend
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit requests
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 5.4 docker-compose.yml
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - C:/Users/mlc/.ollama:/root/.ollama   # Windows 路径需用正斜杠
    restart: unless-stopped
    command: serve

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    volumes:
      - C:/Users/mlc/.cache/huggingface/hub:/root/.cache/huggingface/hub   # 共享 embedding 缓存
    depends_on:
      - ollama
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
    restart: unless-stopped
```

### 5.5 构建与启动
```bash
docker-compose build
docker-compose up -d
```

### 5.6 遇到的典型 Docker 问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `failed to solve: process "/bin/sh -c pip install ..."` | pip 安装超时或版本不存在 | 使用国内镜像源，删去不存在的版本号（如 `langchain==1.0.39` 改为 `langchain==0.2.16`） |
| `OSError: We couldn't connect to 'https://hf-mirror.com'` | 容器内没有 embedding 模型缓存，且网络无法访问 HuggingFace | 挂载宿主机的缓存目录 `C:/Users/mlc/.cache/huggingface/hub:/root/.cache/huggingface/hub` |
| `Error response from daemon: ports are not available: bind: Only one usage...` | 宿主机 Ollama 或 uvicorn 已占用 11434/8000 端口 | 停止宿主机上的相应服务，或修改 docker-compose 中的映射端口 |
| `services must be a mapping` | YAML 缩进错误，`services` 下面的服务名未缩进 | 确保所有服务名前有两个空格 |
| `additional properties 'backend' not allowed` | `backend` 写在了 `services` 外部 | 确保所有服务都在 `services:` 下面 |

---

## 六、多轮对话记忆的添加

### 6.1 后端实现（在 api.py 中添加）
```python
from typing import Dict, List
chat_histories: Dict[str, List[dict]] = {}
MAX_HISTORY_ROUNDS = 5

# 在 prompt 模板中加入 {history} 占位符
# 在 /chat 端点中：
history = chat_histories.get(request.conversation_id, [])
recent = history[-(MAX_HISTORY_ROUNDS*2):]
history_text = "\n".join([...])   # 拼接历史
response = agent_executor.invoke({"input": request.question, "history": history_text})
# 保存本轮对话
history.append(...)
chat_histories[request.conversation_id] = history
```

### 6.2 遇到问题
- **`NameError: name 'llm' is not defined`**：添加代码时误将 `llm` 定义移动到了后面。调整回正确顺序即可。
- **历史过长导致 token 超限**：限制 `MAX_HISTORY_ROUNDS` 为 5 轮，或使用摘要记忆。

---

## 七、常见问题 FAQ

### Q1：启动后前端报“连接后端失败”
- 首先用 `curl http://localhost:8000/health` 测试后端是否正常。
- 若后端返回正常，检查前端 `app.py` 中的 `API_URL`，确保其值为 `http://backend:8000`（Docker 内）或 `http://localhost:8000`（宿主机）。
- 若后端无响应，执行 `docker-compose logs --tail=30 backend` 查看错误。

### Q2：Agent 不调用知识库工具，直接凭记忆回答
修改 prompt，在开头明确要求：“请优先使用 KnowledgeBase 工具检索信息，不要凭内部知识回答，除非确实没有相关内容。”

### Q3：参考来源不显示或显示不全
- 确认 `AgentExecutor` 中 `return_intermediate_steps=True`。
- 在 `/chat` 端点中正确提取 `intermediate_steps`，并过滤 `action.tool == "KnowledgeBase"`。
- 截断长度可调整（`observation[:500]`）。

### Q4：模型回答速度太慢
- 换用量化版：`qwen2.5:7b-q4_K_M` 或 `qwen2.5:3b`。
- 为 Docker 分配更多 CPU/内存（Docker Desktop → Settings → Resources）。
- 减少 `max_iterations` 到 5，降低 Agent 循环次数。

### Q5：Docker 构建时 pip 安装失败
- 使用国内镜像：在 `Dockerfile` 中添加 `RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。
- 确认 `requirements.txt` 中的版本号真实存在（可先 `pip install 包名==版本号` 验证）。

### Q6：多轮对话不记忆
- 检查 `chat_histories` 是否正确更新（观察日志 `收到问题: ... (会话: ...)` 是否一致）。
- 确保请求中 `conversation_id` 保持不变（前端默认 `"default"`）。

---

## 八、项目文件清单
```
.
├── api.py                 # FastAPI 后端（含 Agent + 多轮记忆）
├── app.py                 # Streamlit 前端
├── ingest.py             # 向量数据库生成器
├── data.txt              # 知识文档
├── chroma_db/            # 向量数据库目录（由 ingest.py 生成）
├── requirements.txt      # Python 依赖
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── .dockerignore
```

---

## 九、快速启动（已有 Docker 的环境）
```bash
git clone <your-repo>
cd contract-agent
docker-compose build
docker-compose up -d
```
然后访问 `http://localhost:8501` 即可使用。

---

## 十、总结
本项目从零搭建了一个企业级 RAG + Agent 系统，涵盖了文档知识库、工具调用、多轮对话、容器化部署等关键能力。通过过程中遇到的问题（版本兼容、网络访问、端口冲突、Docker 构建等）及解决方案，可以快速积累 AI 工程化实战经验。