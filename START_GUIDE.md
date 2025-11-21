# 🚀 启动指南

## 方式一：一键启动（推荐）✨

最简单的方式，自动启动前后端：

```bash
python scripts/start_services.py
```

启动后会显示：
- 📱 前端地址: http://localhost:3000
- 🔧 后端API: http://localhost:8000
- 🎤 实时语音识别: ws://localhost:8000/ws/realtime-speech

按 `Ctrl+C` 可以停止所有服务。

---

## 方式二：分别启动

### 1. 启动后端服务器

**方法A：使用模块方式（推荐）**
```bash
python -m app.server
```

**方法B：直接运行文件**
```bash
python app/server.py
```

后端会在 `http://localhost:8000` 启动

### 2. 启动前端服务器

打开**新的终端窗口**，然后：

```bash
cd web
node server.js
```

前端会在 `http://localhost:3000` 启动

---

## 📋 启动前检查清单

### ✅ 1. 环境准备
```bash
# 确保已安装Python依赖
pip install -r requirements.txt

# 确保已安装Node.js依赖
cd web
npm install
cd ..
```

### ✅ 2. 配置API Key
确保已设置环境变量：
```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key"

# Windows CMD
set DASHSCOPE_API_KEY=your_api_key

# Linux/Mac
export DASHSCOPE_API_KEY="your_api_key"
```

或者在项目根目录创建 `.env` 文件：
```
DASHSCOPE_API_KEY=your_api_key
```

### ✅ 3. 导入知识库（首次使用）
```bash
python scripts/ingest_documents.py
```

---

## 🌐 访问系统

启动成功后，在浏览器中访问：
- **前端界面**: http://localhost:3000
- **后端API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/health

---

## 🎤 使用语音功能

1. 在浏览器中打开 http://localhost:3000
2. 点击「语音模式」按钮
3. 点击「录音」按钮开始说话
4. 系统会实时识别语音并显示在输入框中
5. 再次点击「录音」按钮结束录音
6. 识别结果会自动发送给RAG系统进行回答

---

## ⚠️ 常见问题

### 问题1：端口被占用
如果8000或3000端口被占用，可以修改：
- **后端端口**：编辑 `app/server.py` 第822行的 `port=8000`
- **前端端口**：编辑 `web/server.js` 中的端口配置

### 问题2：前端无法连接后端
检查：
1. 后端是否正常启动（访问 http://localhost:8000/api/health）
2. 前端代码中的WebSocket地址是否正确（应该是 `ws://localhost:8000`）

### 问题3：语音识别不工作
检查：
1. 浏览器是否允许麦克风权限
2. `DASHSCOPE_API_KEY` 是否正确配置
3. 网络连接是否正常（需要访问阿里云API）

---

## 🔧 开发模式

### 后端热重载
后端默认开启了热重载，修改代码后会自动重启。

### 前端开发
如果需要前端热重载，可以使用：
```bash
cd web
npm run dev  # 如果有配置的话
```

---

## 📝 日志查看

### 后端日志
后端日志会直接输出到终端，包括：
- WebSocket连接信息
- 语音识别结果
- RAG检索和生成过程

### 前端日志
打开浏览器开发者工具（F12）查看控制台日志。


