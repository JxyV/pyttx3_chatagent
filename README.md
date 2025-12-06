# 湖北博物馆智能问答系统

## 🎯 系统介绍

本系统提供两种使用方式：**终端版本**和**网页版本**，支持文本和语音交互。

### 🖥️ 终端版本
- 命令行交互界面
- 支持语音输入/输出
- 适合开发调试和服务器环境

### 🌐 网页版本  
- 现代化Web界面
- 实时语音识别和合成
- 适合用户交互和演示展示

## ⚠️ 重要更新

### 🎙️ 最新: 本地 FunASR Paraformer STT（2025-12-05）

- ✅ **本地部署**: 默认使用 IIC Paraformer Online 模型 `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`
- ✅ **低延迟流式**: 支持 16kHz 单声道 PCM，提供一次性与流式转写
- ✅ **无云依赖**: 不再需要 DashScope/Gummy API Key，全部在本地 GPU/CPU 运行
- 🔧 **运行设备**: 默认 `cuda:0`，无 GPU 可改为 `cpu`

### 🗄️ 已迁移到Milvus Lite

本项目已从 Chroma 数据库迁移到 **Milvus Lite**，获得更好的性能和扩展性！

- 🚀 **新用户**: 直接按照下面的快速开始步骤操作
- 🔄 **现有用户**: 请查看 [迁移指南](MILVUS_MIGRATION_GUIDE.md) 进行数据迁移
- 📚 **详细文档**: 
  - [系统使用说明](SYSTEM_USAGE.md) - **新用户必读**
  - [Paraformer前端集成](PARAFORMER_FRONTEND_INTEGRATION.md) - **语音识别说明**
  - [快速开始指南](QUICK_START_MILVUS.md)
  - [迁移总结](MIGRATION_SUMMARY.md)
  - [Chroma vs Milvus对比](CHROMA_VS_MILVUS.md)

---

## 🚀 快速启动

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 2. （可选）配置环境变量
- 本地 STT 已默认启用，无需 API Key
- 可通过 `.env` 配置 TTS 语速/音量等参数（如 `TTS_RATE`、`TTS_VOLUME`、`TTS_VOICE_ID`）

### 3. 导入知识库
```bash
python scripts/ingest_documents.py
```

### 4. 启动系统

#### 🖥️ 终端版本
```bash
python scripts/run_voice_rag.py
```

#### 🌐 网页版本
```bash
# 一键启动（推荐）
python scripts/start_services.py

# 或分别启动
python -m app.server  # 后端
cd web && node server.js  # 前端
```

然后访问：http://localhost:3000

---

## 📚 详细使用说明

请查看 [系统使用说明](SYSTEM_USAGE.md) 了解两个版本的详细功能和使用方法。

---

## 📋 系统特性

### 核心功能
- 📚 **知识库检索**: 基于湖北博物馆文档的智能问答
- 🎤 **语音交互**: 支持语音输入和语音输出
- 🔍 **RAG检索**: 检索增强生成，提供准确答案和引用
- 💬 **多模态交互**: 文本和语音双重交互方式

### 技术栈
- **向量数据库**: Milvus Lite (已从Chroma迁移)
- **嵌入模型**: sentence-transformers
- **大语言模型**: Ollama (本地) 或 OpenAI
- **语音识别**: 本地 FunASR Paraformer (IIC)
- **语音合成**: 本地 pyttsx3 TTS（可替换其他方案）
- **前端框架**: 原生JavaScript + WebSocket
- **后端框架**: FastAPI + WebSocket

### Models
- Embeddings (choose via `.env`):
  - sentence-transformers: e.g., `BAAI/bge-small-en-v1.5` (English) / `BAAI/bge-m3` (multilingual/Chinese)
  - Ollama Embeddings: set `EMBEDDING_BACKEND=ollama` and optionally `EMBEDDING_MODEL=bge-m3`
- LLM Backends:
  - Ollama: set `LLM_BACKEND=ollama` and `OLLAMA_MODEL` (e.g., `qwen2.5:7b`, `llama3.1:8b-instruct`)
  - OpenAI: set `LLM_BACKEND=openai` and `OPENAI_API_KEY`
- STT Models:
  - **默认**: IIC FunASR Paraformer Online `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`（16kHz，device=cuda:0，可改cpu）

### Ollama Notes
- Install Ollama: see `https://ollama.ai/`
- Pull models, e.g.:
```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b-instruct
ollama pull bge-m3
```
- Ensure the service is running at `http://localhost:11434` (default). If not, set `OLLAMA_BASE_URL` in `.env`.

### Environment Variables (.env)
- EMBEDDING_BACKEND: `sentence-transformers` | `ollama`
- EMBEDDING_MODEL: model name, e.g. `BAAI/bge-small-en-v1.5` or `BAAI/bge-m3`
- LLM_BACKEND: `ollama` | `openai`
- OLLAMA_MODEL: e.g., `qwen2.5:7b`
- OPENAI_API_KEY: your key if using OpenAI
- CHUNK_SIZE: default 800
- CHUNK_OVERLAP: default 120
- TOP_K: default 4
- DOCS_DIR: default `docs`
- CHROMA_PERSIST_DIR: default `.chroma`
- **CHROMA_COLLECTION_NAME**: default `rag_docs` (自定义数据库名称)

### Workflow
1. `ingest.py` loads files, chunks, embeds, and writes to Chroma
2. `rag_chain.py` builds the retriever and LCEL RAG chain
3. `cli.py` and `server.py` call the chain to answer questions and return citations

### FAQ
- No results found?
  - You will get a polite message indicating uncertainty if retrieval returns nothing.
- Switching backends?
  - Modify `.env` only; code reads configuration dynamically.
- PDF page numbers?
  - Citations include `filename` and `page` when available; otherwise the `chunk_id`.

### 自定义Chroma数据库名称

**问题**: Chroma数据库名称显示为乱码或默认名称  
**解决**: 通过环境变量自定义数据库名称

1. **在 `.env` 文件中设置**：
```bash
# 自定义数据库名称
CHROMA_COLLECTION_NAME=my_knowledge_base
CHROMA_PERSIST_DIR=.chroma
```

2. **重新构建向量库**：
```bash
python ingest.py
```

3. **验证数据库名称**：
```bash
# 查看 .chroma 目录下的文件
ls -la .chroma/
```

**支持的名称格式**：
- 英文: `my_knowledge_base`
- 中文: `我的知识库` 
- 数字: `knowledge_base_2024`
- 下划线: `museum_docs`

### Development
- Python 3.10+
- Keep docs small for quick local testing.
