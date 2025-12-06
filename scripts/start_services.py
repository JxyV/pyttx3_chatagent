#!/usr/bin/env python3
"""
启动完整的博物馆RAG系统（包含实时语音功能）
"""

import subprocess
import sys
import time
import signal
import os

def start_backend_server():
    """启动WebSocket后端服务器（包含实时语音功能）"""
    print("🚀 启动WebSocket后端服务器（包含实时语音识别和TTS）...")
    return subprocess.Popen([
        sys.executable, "-m", "app.server"
    ], cwd=os.getcwd())

def check_frontend_dependencies():
    """检查前端依赖是否已安装"""
    web_dir = os.path.join(os.getcwd(), "web")
    node_modules = os.path.join(web_dir, "node_modules")
    if not os.path.exists(node_modules):
        print("⚠️  前端依赖未安装，正在安装...")
        print("   这可能需要几分钟时间，请耐心等待...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=web_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ 前端依赖安装失败: {result.stderr}")
            return False
        print("✅ 前端依赖安装完成")
    return True

def start_frontend_server():
    """启动Node.js前端服务器"""
    if not check_frontend_dependencies():
        print("❌ 无法启动前端服务器，请手动运行: cd web && npm install")
        return None
    
    print("🌐 启动前端服务器...")
    return subprocess.Popen([
        "node", "server.js"
    ], cwd=os.path.join(os.getcwd(), "web"))

def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n🛑 正在关闭所有服务...")
    for process in processes:
        if process.poll() is None:
            process.terminate()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    processes = []
    
    try:
        print("=" * 60)
        print("🏛️  湖北博物馆智能问答系统 - 实时语音版")
        print("=" * 60)
        print()
        
        # 启动后端服务器（包含实时语音功能）
        backend_process = start_backend_server()
        processes.append(backend_process)
        time.sleep(3)  # 等待后端完全启动
        
        # 启动前端服务器
        frontend_process = start_frontend_server()
        if frontend_process:
            processes.append(frontend_process)
            time.sleep(2)
        else:
            print("⚠️  前端服务器启动失败，但后端已正常运行")
        
        print()
        print("=" * 60)
        print("✅ 所有服务已启动!")
        print("=" * 60)
        print()
        print("📱 前端地址: http://localhost:3000")
        print("🔧 后端API: http://localhost:8000")
        print("🎤 实时语音识别: ws://localhost:8000/ws/realtime-speech")
        print("🔊 实时语音合成: ws://localhost:8000/ws/tts-stream")
        print()
        print("💡 提示：")
        print("   1. 在浏览器中访问 http://localhost:3000")
        print("   2. 点击「语音模式」按钮启用语音交互")
        print("   3. 直接说话，系统会实时识别并自动回答")
        print()
        print("⚠️  提示：默认使用本地 FunASR STT（16kHz，device=cuda:0，可改cpu）")
        print()
        print("按 Ctrl+C 停止所有服务")
        print("=" * 60)
        print()
        
        # 等待所有进程
        for process in processes:
            process.wait()
            
    except Exception as e:
        print(f"\n❌ 启动服务时发生错误: {e}")
        for process in processes:
            if process.poll() is None:
                process.terminate()
