"""
环境变量配置助手
帮助用户快速创建 .env 文件
"""
import os
from pathlib import Path


def create_env_file():
    """创建 .env 文件"""
    print("=" * 70)
    print("🔧 环境变量配置助手")
    print("=" * 70)
    print()
    
    # 检查是否已存在 .env 文件
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  检测到已存在 .env 文件")
        choice = input("是否要覆盖? (y/n): ").strip().lower()
        if choice != 'y':
            print("❌ 已取消")
            return
        print()
    
    print("请选择配置模式：")
    print("1. 最小配置（仅本地模型，免费）")
    print("2. 标准配置（本地 STT + 云端 TTS）")
    print("3. 完整配置（手动输入所有配置）")
    print()
    
    choice = input("请选择 (1-3): ").strip()
    print()
    
    if choice == "1":
        config = generate_minimal_config()
    elif choice == "2":
        config = generate_standard_config()
    elif choice == "3":
        config = generate_full_config()
    else:
        print("❌ 无效的选择")
        return
    
    # 写入 .env 文件
    with open(".env", "w", encoding="utf-8") as f:
        f.write(config)
    
    print()
    print("=" * 70)
    print("✅ .env 文件创建成功！")
    print("=" * 70)
    print()
    print("📍 文件位置:", env_file.absolute())
    print()
    print("🚀 下一步：")
    print("   1. 如需修改配置，直接编辑 .env 文件")
    print("   2. 运行 'python test_realtime_streaming.py' 测试流式识别")
    print("   3. 运行 'python multimodal_rag.py' 启动系统")
    print()


def generate_minimal_config() -> str:
    """生成最小配置"""
    print("📝 最小配置模式")
    print("   - ✅ 本地 Paraformer 语音识别（免费）")
    print("   - ✅ 实时流式识别")
    print("   - ❌ 无 TTS 语音合成")
    print()
    
    return """# ==================== 最小配置（仅本地模型） ====================

# 语音识别配置
STT_TYPE=paraformer
PARAFORMER_STREAMING=true
PARAFORMER_MODEL=damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

# 日志配置
LOG_LEVEL=INFO

# 注意：此配置不包含 TTS，无法使用语音合成功能
# 如需 TTS，请添加 DASHSCOPE_API_KEY 并设置 TTS_MODEL
"""


def generate_standard_config() -> str:
    """生成标准配置（需要用户输入 API Key）"""
    print("📝 标准配置模式")
    print("   - ✅ 本地 Paraformer 语音识别（免费）")
    print("   - ✅ 实时流式识别")
    print("   - ✅ 阿里云 TTS 语音合成（需要 API Key）")
    print()
    
    # 询问 API Key
    api_key = input("请输入阿里云 DashScope API Key（按回车跳过）: ").strip()
    print()
    
    if not api_key:
        print("⚠️  未输入 API Key，将使用占位符")
        print("   请稍后编辑 .env 文件添加真实的 API Key")
        api_key = "your_dashscope_api_key_here"
    
    # 询问 TTS 音色
    print("请选择 TTS 音色：")
    voices = {
        "1": ("Cherry", "芊悦（女，标准普通话）"),
        "2": ("Ethan", "晨煦（男，标准普通话）"),
        "3": ("Nofish", "不吃鱼（女，活泼）"),
        "4": ("Jennifer", "詹妮弗（女，英语）"),
    }
    
    for key, (name, desc) in voices.items():
        print(f"{key}. {name} - {desc}")
    
    voice_choice = input("\n请选择 (1-4，按回车使用默认): ").strip() or "1"
    voice_name = voices.get(voice_choice, ("Cherry", "芊悦"))[0]
    print()
    
    return f"""# ==================== 标准配置（本地 STT + 云端 TTS） ====================

# 阿里云 API 配置
DASHSCOPE_API_KEY={api_key}

# 语音识别配置（本地模型）
STT_TYPE=paraformer
PARAFORMER_STREAMING=true
PARAFORMER_MODEL=damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

# 语音合成配置（阿里云）
TTS_MODEL=cosyvoice-v1
TTS_VOICE={voice_name}

# 日志配置
LOG_LEVEL=INFO
"""


def generate_full_config() -> str:
    """生成完整配置（用户手动输入）"""
    print("📝 完整配置模式")
    print()
    
    config = {}
    
    # API Key
    config['DASHSCOPE_API_KEY'] = input(
        "阿里云 DashScope API Key（按回车跳过）: "
    ).strip() or "your_dashscope_api_key_here"
    
    # STT 配置
    print("\n语音识别配置：")
    print("1. paraformer（本地，免费）")
    print("2. gummy（阿里云，需要 API Key）")
    stt_choice = input("选择 STT 类型 (1-2): ").strip()
    config['STT_TYPE'] = "paraformer" if stt_choice == "1" else "gummy"
    
    if config['STT_TYPE'] == "paraformer":
        streaming = input("启用流式识别? (y/n): ").strip().lower()
        config['PARAFORMER_STREAMING'] = "true" if streaming == "y" else "false"
    
    # TTS 配置
    print("\nTTS 音色（可选: Cherry, Ethan, Nofish, Jennifer）:")
    config['TTS_VOICE'] = input("输入音色名称（按回车使用 Cherry）: ").strip() or "Cherry"
    
    # 日志级别
    print("\n日志级别（DEBUG, INFO, WARNING, ERROR）:")
    config['LOG_LEVEL'] = input("输入日志级别（按回车使用 INFO）: ").strip() or "INFO"
    
    print()
    
    # 生成配置文本
    return f"""# ==================== 完整配置 ====================

# 阿里云 API 配置
DASHSCOPE_API_KEY={config['DASHSCOPE_API_KEY']}

# 语音识别配置
STT_TYPE={config['STT_TYPE']}
{'PARAFORMER_STREAMING=' + config.get('PARAFORMER_STREAMING', 'true') if config['STT_TYPE'] == 'paraformer' else '# PARAFORMER_STREAMING=true'}
PARAFORMER_MODEL=damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

# 语音合成配置
TTS_MODEL=cosyvoice-v1
TTS_VOICE={config['TTS_VOICE']}

# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DATABASE=museum_rag

# 日志配置
LOG_LEVEL={config['LOG_LEVEL']}

# 其他配置
VECTOR_DB=milvus
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
"""


def show_existing_config():
    """显示现有配置"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ 未找到 .env 文件")
        return
    
    print("=" * 70)
    print("📄 当前 .env 配置")
    print("=" * 70)
    print()
    
    with open(env_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 隐藏敏感信息
    for line in content.split("\n"):
        if "API_KEY" in line and "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            if value and not value.startswith("your_"):
                print(f"{key}=***HIDDEN***")
            else:
                print(line)
        else:
            print(line)
    
    print()


def main():
    """主函数"""
    print()
    
    while True:
        print("=" * 70)
        print("环境变量配置助手")
        print("=" * 70)
        print()
        print("1. 创建/更新 .env 文件")
        print("2. 查看当前配置")
        print("3. 查看配置文档")
        print("4. 退出")
        print()
        
        choice = input("请选择 (1-4): ").strip()
        print()
        
        if choice == "1":
            create_env_file()
        elif choice == "2":
            show_existing_config()
        elif choice == "3":
            print("📚 配置文档位置：")
            print("   - ENV_CONFIG.md - 环境变量配置指南")
            print("   - STREAMING_ASR_USAGE.md - 流式识别使用指南")
            print("   - env_paraformer_template.txt - 配置模板")
            print()
        elif choice == "4":
            print("👋 再见！")
            break
        else:
            print("❌ 无效的选择")
        
        input("\n按回车继续...")
        print("\n" * 2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

