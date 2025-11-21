#!/usr/bin/env python3
"""
启动使用Paraformer本地模型的前端系统
"""
import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查依赖是否已安装"""
    logger.info("检查依赖...")
    
    dependencies = {
        'funasr': 'FunASR',
        'modelscope': 'ModelScope',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'websockets': 'WebSockets',
        'numpy': 'NumPy'
    }
    
    missing = []
    for package, name in dependencies.items():
        try:
            __import__(package)
            logger.info(f"✅ {name} 已安装")
        except ImportError:
            logger.error(f"❌ {name} 未安装")
            missing.append(package)
    
    if missing:
        logger.error(f"\n缺少依赖包: {', '.join(missing)}")
        logger.info("\n请运行以下命令安装:")
        logger.info(f"pip install {' '.join(missing)}")
        return False
    
    logger.info("✅ 所有依赖已安装")
    return True


def check_dashscope_api():
    """检查DashScope API配置"""
    logger.info("\n检查DashScope API配置...")
    
    try:
        import dashscope
        from dashscope.audio.asr import Recognition
        
        # 检查API Key
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("❌ DASHSCOPE_API_KEY环境变量未设置")
            logger.info("\n请设置API Key:")
            logger.info("Windows: setx DASHSCOPE_API_KEY \"your_api_key\"")
            logger.info("Linux/Mac: export DASHSCOPE_API_KEY=\"your_api_key\"")
            return False
        
        logger.info(f"✅ DASHSCOPE_API_KEY已配置: {api_key[:10]}...")
        
        # 测试API连接
        logger.info("测试API连接...")
        recognition = Recognition(
            model='paraformer-realtime-8k-v2',
            format='wav',
            sample_rate=8000,
            callback=None  # 添加必需的callback参数
        )
        logger.info("✅ DashScope API连接正常")
        return True
        
    except ImportError:
        logger.error("❌ dashscope未安装")
        logger.info("\n请安装依赖:")
        logger.info("pip install dashscope")
        return False
    except Exception as e:
        logger.error(f"❌ API测试失败: {e}")
        logger.info("\n可能原因:")
        logger.info("1. API Key无效")
        logger.info("2. 网络连接问题")
        logger.info("3. ![1760950502170](image/start_paraformer_frontend/1760950502170.png)")
        return False


def start_backend():
    """启动后端服务器"""
    logger.info("\n" + "="*60)
    logger.info("启动后端WebSocket服务器...")
    logger.info("="*60)
    
    try:
        # 启动websocket_server.py - 显示所有输出
        process = subprocess.Popen(
            [sys.executable, "websocket_server.py"],
            stdout=None,  # 直接显示到控制台
            stderr=None,  # 直接显示到控制台
            universal_newlines=True
        )
        
        # 等待服务器启动
        logger.info("等待服务器启动...")
        time.sleep(5)  # 增加等待时间，确保服务器完全启动
        
        if process.poll() is not None:
            logger.error("❌ 后端服务器启动失败，进程已退出")
            logger.error("   请检查上面的错误信息")
            return None
        
        logger.info("✅ 后端服务器启动成功")
        logger.info("   地址: http://localhost:8000")
        logger.info("   WebSocket: ws://localhost:8000/ws/realtime-speech")
        
        return process
        
    except Exception as e:
        logger.error(f"❌ 启动后端服务器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def start_frontend():
    """启动前端服务器"""
    logger.info("\n" + "="*60)
    logger.info("启动前端服务器...")
    logger.info("="*60)
    
    try:
        # 检查frontend目录
        if not os.path.exists("frontend"):
            logger.error("❌ frontend目录不存在")
            return None
        
        # 检查node_modules
        if not os.path.exists("frontend/node_modules"):
            logger.info("安装前端依赖...")
            subprocess.run(
                ["npm", "install"],
                cwd="frontend",
                check=True
            )
        
        # 启动前端服务器，使用端口3001避免冲突
        process = subprocess.Popen(
            ["node", "server.js"],
            cwd="frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, "PORT": "3001"}
        )
        
        # 等待服务器启动
        logger.info("等待前端服务器启动...")
        time.sleep(2)
        
        if process.poll() is not None:
            logger.error("❌ 前端服务器启动失败")
            # 读取错误输出
            try:
                output, _ = process.communicate(timeout=1)
                if output:
                    logger.error(f"错误输出: {output}")
            except:
                pass
            return None
        
        logger.info("✅ 前端服务器启动成功")
        logger.info("   地址: http://localhost:3001")
        
        return process
        
    except FileNotFoundError:
        logger.error("❌ Node.js未安装或未添加到PATH")
        logger.info("   请安装Node.js: https://nodejs.org/")
        return None
    except Exception as e:
        logger.error(f"❌ 启动前端服务器失败: {e}")
        return None


def main():
    """主函数"""
    print("\n" + "="*60)
    print("阿里云API语音识别前端系统启动器")
    print("="*60 + "\n")
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 请先安装缺少的依赖")
        return
    
    # 检查DashScope API
    if not check_dashscope_api():
        print("\n❌ DashScope API配置有问题，请检查API Key设置")
        return
    
    # 启动后端
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ 后端启动失败，退出")
        return
    
    # 启动前端
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ 前端启动失败，停止后端")
        backend_process.terminate()
        return
    
    # 显示使用说明
    print("\n" + "="*60)
    print("🎉 系统启动成功!")
    print("="*60)
    print("\n📝 使用说明:")
    print("   1. 打开浏览器访问: http://localhost:3001")
    print("   2. 点击麦克风按钮 🎤 开始语音识别")
    print("   3. 说话时会实时显示识别结果")
    print("   4. 停止录音后识别结果会自动发送给AI")
    print("\n💡 技术特点:")
    print("   ✅ 使用阿里云paraformer-realtime-8k-v2 API")
    print("   ✅ 高准确率，支持8kHz音频")
    print("   ✅ 稳定可靠，自动负载均衡")
    print("   ✅ 支持高并发，易于扩展")
    print("\n⚠️ 按 Ctrl+C 停止所有服务")
    print("="*60 + "\n")
    
    try:
        # 保持进程运行
        while True:
            time.sleep(1)
            
            # 检查进程是否还在运行
            if backend_process.poll() is not None:
                logger.error("后端进程意外退出")
                break
            if frontend_process.poll() is not None:
                logger.error("前端进程意外退出")
                break
                
    except KeyboardInterrupt:
        print("\n\n正在停止服务...")
        
        # 停止前端
        if frontend_process:
            frontend_process.terminate()
            logger.info("✅ 前端服务器已停止")
        
        # 停止后端
        if backend_process:
            backend_process.terminate()
            logger.info("✅ 后端服务器已停止")
        
        print("\n👋 再见!")


if __name__ == "__main__":
    main()

