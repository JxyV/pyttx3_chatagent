#!/usr/bin/env python3
"""
重组项目结构脚本
"""
import os
import shutil
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent

def create_directories():
    """创建新的目录结构"""
    dirs = [
        "app",
        "app/conversation",
        "app/rag",
        "app/services",
        "scripts",
        "data",
        "data/db",
        "web",
        "web/src"
    ]
    
    for dir_path in dirs:
        (ROOT / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] 创建目录: {dir_path}")
    
    # 创建__init__.py文件
    init_files = [
        "app/__init__.py",
        "app/conversation/__init__.py",
        "app/rag/__init__.py",
        "app/services/__init__.py"
    ]
    
    for init_file in init_files:
        (ROOT / init_file).touch(exist_ok=True)
        print(f"[OK] 创建文件: {init_file}")

def move_files():
    """移动文件到新位置"""
    moves = [
        # 核心文件到app/
        ("config.py", "app/config.py"),
        ("websocket_server.py", "app/server.py"),
        
        # RAG相关到app/rag/
        ("rag_chain.py", "app/rag/chain.py"),
        
        # 服务相关到app/services/
        ("voice_interface.py", "app/services/voice.py"),
        ("milvus_manager.py", "app/services/milvus.py"),
        ("milvus_standalone_manager.py", "app/services/milvus_standalone.py"),
        ("milvus_cli.py", "app/services/milvus_cli.py"),
        
        # 脚本文件到scripts/
        ("ingest.py", "scripts/ingest_documents.py"),
        ("multimodal_rag.py", "scripts/run_voice_rag.py"),
        ("start_realtime_services.py", "scripts/start_services.py"),
        ("setup_env.py", "scripts/setup_env.py"),
        ("simple_voice_test.py", "scripts/stt_realtime_demo.py"),
        
        # 数据目录
        ("data_db", "data/db"),
    ]
    
    for src, dst in moves:
        src_path = ROOT / src
        dst_path = ROOT / dst
        
        if src_path.exists():
            if src_path.is_dir():
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.move(str(src_path), str(dst_path))
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
            print(f"[OK] 移动: {src} -> {dst}")
        else:
            print(f"[WARN] 文件不存在: {src}")

def reorganize_frontend():
    """重组前端文件"""
    frontend_src = ROOT / "frontend"
    web_dst = ROOT / "web"
    
    if not frontend_src.exists():
        print("⚠️  frontend目录不存在")
        return
    
    # 移动public目录内容到web/src
    public_src = frontend_src / "public"
    if public_src.exists():
        for item in public_src.iterdir():
            if item.name != "uploads":  # 保留uploads目录在原位置
                dst = web_dst / "src" / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(item), str(dst))
                else:
                    shutil.move(str(item), str(dst))
                print(f"[OK] 移动前端文件: {item.name}")
    
    # 移动package.json等文件到web/
    for item in ["package.json", "package-lock.json", "server.js"]:
        src = frontend_src / item
        if src.exists():
            shutil.move(str(src), str(web_dst / item))
            print(f"[OK] 移动: {item}")

def main():
    """主函数"""
    print("=" * 50)
    print("开始重组项目结构...")
    print("=" * 50)
    
    create_directories()
    print()
    
    move_files()
    print()
    
    reorganize_frontend()
    print()
    
    print("=" * 50)
    print("[OK] 项目结构重组完成！")
    print("=" * 50)
    print("\n[注意] 请检查以下事项：")
    print("1. 请检查所有导入路径是否正确")
    print("2. 请更新配置文件中的路径引用")
    print("3. 请测试系统是否正常运行")

if __name__ == "__main__":
    main()

