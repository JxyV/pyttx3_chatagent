# 项目结构说明

## 📁 目录结构

```
Museum_RAGmilvus/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖
├── app/                      # 主应用代码
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── server.py            # WebSocket服务器
│   ├── conversation/        # 对话管理模块
│   │   └── __init__.py
│   ├── rag/                # RAG检索增强生成
│   │   ├── __init__.py
│   │   └── chain.py        # RAG链构建
│   └── services/           # 服务模块
│       ├── __init__.py
│       ├── voice.py         # 语音服务（STT/TTS）
│       ├── milvus.py        # Milvus数据库管理
│       ├── milvus_standalone.py  # Milvus独立模式
│       └── milvus_cli.py    # Milvus命令行工具
├── scripts/                 # 脚本文件
│   ├── ingest_documents.py  # 文档导入
│   ├── run_voice_rag.py    # 运行语音RAG
│   ├── start_services.py    # 启动服务
│   ├── setup_env.py        # 环境配置
│   └── stt_realtime_demo.py # STT实时演示
├── data/                    # 数据目录
│   └── db/                 # 数据库存储
│       ├── milvus_db/      # Milvus数据
│       └── chroma_db/      # Chroma数据（旧）
├── docs/                    # 知识库文档
├── web/                     # 前端代码
│   ├── package.json
│   ├── server.js           # 前端服务器
│   └── src/                # 前端源码
│       ├── index.html
│       ├── app.js
│       ├── styles.css
│       └── app_debug.js
└── reorganize_project.py    # 项目重组脚本（可删除）
```

## 🔧 主要模块说明

### app/ - 主应用
- **config.py**: 系统配置管理，包含数据库、模型、语音等配置
- **server.py**: FastAPI WebSocket服务器，处理实时通信

### app/rag/ - RAG模块
- **chain.py**: 构建RAG链，包括检索器和LLM

### app/services/ - 服务模块
- **voice.py**: 语音识别（STT）和语音合成（TTS）服务
- **milvus.py**: Milvus向量数据库管理
- **milvus_standalone.py**: Milvus独立模式管理
- **milvus_cli.py**: Milvus命令行工具

### scripts/ - 脚本
- **ingest_documents.py**: 导入文档到向量数据库
- **run_voice_rag.py**: 运行终端版本的语音RAG
- **start_services.py**: 一键启动前后端服务

### web/ - 前端
- **src/**: 前端源码（HTML、JavaScript、CSS）
- **server.js**: Node.js前端服务器

## 🚀 快速开始

### 1. 导入文档
```bash
python scripts/ingest_documents.py
```

### 2. 启动终端版本
```bash
python scripts/run_voice_rag.py
```

### 3. 启动Web版本
```bash
python scripts/start_services.py
```

或分别启动：
```bash
# 后端
python -m app.server

# 前端
cd web && node server.js
```

## 📝 导入路径说明

### 在app/目录内的文件
```python
from app.config import get_rag_config
from app.rag.chain import build_chain
from app.services.voice import VoiceInterface
```

### 在scripts/目录内的文件
```python
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_rag_config
from app.rag.chain import build_chain
```

## 🔄 迁移说明

如果从旧结构迁移，请确保：
1. 所有导入路径已更新
2. 配置文件中的路径已更新
3. 前端路径已更新（frontend -> web）

