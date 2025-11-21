"""使用 gummy-realtime-v1 对本地 PCM 文件进行实时识别示例"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.services.stt_client import GummyRealtimeSTT


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description="gummy-realtime-v1 实时识别示例（本地 PCM 文件）")
    parser.add_argument("--audio", default="./your_audio_file.pcm", help="PCM16/16k 单声道音频路径")
    parser.add_argument("--model", default="gummy-realtime-v1", help="识别模型名称")
    args = parser.parse_args()

    setup_logging()

    # 检查API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ 错误: 找不到音频文件：{audio_path}")
        sys.exit(1)

    print(f"📁 音频文件: {audio_path}")
    print(f"🎤 模型: {args.model}")
    print("🔄 开始识别...")
    print()

    try:
        stt = GummyRealtimeSTT(api_key=api_key, model=args.model)
        result = stt.transcribe(audio_path)
        
        print()
        print("=" * 50)
        print("✅ 识别完成！")
        print("=" * 50)
        print(f"识别结果: {result}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
