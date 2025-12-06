"""使用本地 FunASR Paraformer 对本地 PCM 文件进行实时识别示例"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.services.voice import IicRealtimeSTT


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description="本地 Paraformer 实时识别示例（PCM16/16k 单声道）")
    parser.add_argument("--audio", default="./your_audio_file.pcm", help="PCM16/16k 单声道音频路径")
    parser.add_argument("--model", default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", help="FunASR模型ID")
    parser.add_argument("--device", default="cuda:0", help="运行设备，例如 cuda:0 或 cpu")
    args = parser.parse_args()

    setup_logging()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ 错误: 找不到音频文件：{audio_path}")
        sys.exit(1)

    print(f"📁 音频文件: {audio_path}")
    print(f"🎤 模型: {args.model}")
    print("🔄 开始识别...")
    print()

    try:
        stt = IicRealtimeSTT(model_id=args.model, device=args.device)
        audio_bytes = audio_path.read_bytes()
        result = stt.transcribe(audio_bytes)
        
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
