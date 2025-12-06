import sys
import time
import queue
import numpy as np
import sounddevice as sd
from funasr import AutoModel

# ========= 基本配置 =========
SAMPLE_RATE = 16000      # FunASR 默认 16k
CHANNELS = 1

# 对应官方流式示例：[0, 10, 5] => 600ms chunk（10 * 60ms） + 300ms lookahead
CHUNK_SIZE = [0, 10, 5]
ENCODER_CHUNK_LOOK_BACK = 4
DECODER_CHUNK_LOOK_BACK = 1

# 960 = 16k * 0.06s，所以 10 * 960 = 600ms
CHUNK_STRIDE = CHUNK_SIZE[1] * 960   # 每块 600ms

MODEL_NAME = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
DEVICE = "cuda:0"   # 没有 GPU 就写 "cpu"

# ========= 新增：静默相关配置 =========
SILENCE_TIMEOUT = 2.0   # 连续 2 秒没有新的识别结果，就认为一句话结束

# ========= 全局状态 =========
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
running = True


def merge_stream_text(full_text: str, last_text: str, new_text: str):
    """
    用“最长公共前缀”把 new_text 合并到 full_text：
    - 如果模型每次给的是“从头到现在的整句”，也能正确只追加新增部分。
    - 如果模型每次给的是“不重叠的小片段”，就退化成简单拼接。
    """
    if not last_text:
        # 第一次直接追加
        return full_text + new_text, new_text

    # 计算 last_text 和 new_text 的最长公共前缀长度
    prefix_len = 0
    for a, b in zip(last_text, new_text):
        if a == b:
            prefix_len += 1
        else:
            break

    append_part = new_text[prefix_len:]
    full_text = full_text + append_part
    last_text = new_text
    return full_text, last_text


def audio_callback(indata, frames, time_info, status):
    """
    sounddevice 的录音回调：
    只负责把数据丢进队列，不在这里做推理（防止阻塞音频线程）
    """
    if status:
        print(f"\n[sounddevice 状态] {status}", file=sys.stderr, flush=True)

    # indata: shape = (frames, channels)，float32
    audio_q.put(indata.copy())


def main():
    global running

    print("加载 FunASR 流式模型中...")
    model = AutoModel(
        model=MODEL_NAME,
        device=DEVICE,
        # 可以关掉自动检查更新，加快启动
        disable_update=True
    )
    print("模型加载完成。")
    print(f"开始实时识别（每块约 {CHUNK_STRIDE / SAMPLE_RATE:.2f} 秒），Ctrl+C 退出...\n")

    cache = {}       # FunASR 流式缓存（必须跨 chunk 共享）
    full_text = ""   # 累计输出的文本（当前这句话）
    last_text = ""   # 上一次模型返回的 text

    # 新增：记录“最后一次有识别文本”的时间
    last_recog_time = None

    # ========== 打开麦克风输入流 ==========　
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_STRIDE,  # 每次回调给我们 600ms
        callback=audio_callback,
    ):
        try:
            while running:
                now = time.time()

                # ========= 检查是否长时间无新文本 =========
                if (
                    full_text              # 当前有累积文本
                    and last_recog_time    # 有过识别时间
                    and (now - last_recog_time) >= SILENCE_TIMEOUT
                ):
                    # 1. 先向模型要“这一句的最终版本”
                    try:
                        res_final = model.generate(
                            input=np.array([], dtype="float32"),  # 空输入，只为了触发 is_final
                            cache=cache,
                            is_final=True,
                            chunk_size=CHUNK_SIZE,
                            encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                            decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                        )
                        if res_final:
                            final_text = res_final[0].get("text", "").strip()
                            if final_text:
                                full_text, last_text = merge_stream_text(full_text, last_text, final_text)
                    except Exception as e:
                        print(f"\n静默收尾 is_final 失败：{e}", file=sys.stderr)

                    # 2. 打印这一句
                    sys.stdout.write("\r")
                    sys.stdout.flush()
                    print("最终一句：", full_text)

                    # 3. ⭐ 关键：重置这一句的所有状态（包括 cache）
                    full_text = ""
                    last_text = ""
                    last_recog_time = None
                    cache = {}   # 告诉模型：下一句从头开始

                    continue  # 回到 while 顶部，不要立刻处理下面这轮的 audio_q.get()

                # ========= 下面是你原来的取音频 + 识别逻辑 =========
                try:
                    data = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                speech_chunk = data[:, 0].astype("float32")

                res = model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=False,
                    chunk_size=CHUNK_SIZE,
                    encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                    decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                )



                if not res:
                    continue

                try:
                    text = res[0].get("text", "").strip()
                except Exception:
                    text = ""

                if not text:
                    continue

                # ========= 有新的识别结果 =========
                # 用“最长公共前缀”合并当前结果
                full_text, last_text = merge_stream_text(full_text, last_text, text)

                # 更新“最后一次识别时间”
                last_recog_time = time.time()

                # 覆盖同一行打印当前正在说的这句话
                sys.stdout.write("\r当前转写：" + full_text)
                sys.stdout.flush()

                # 稍微睡一下，避免疯狂刷屏（可选）
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，准备收尾...")
        finally:
            running = False

    # ========== 收尾：再强制输出最后一截 =========
    try:
        res = model.generate(
            input=np.array([], dtype="float32"),
            cache=cache,
            is_final=True,
            chunk_size=CHUNK_SIZE,
            encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
            decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
        )
        if res:
            final_text = res[0].get("text", "").strip()
            if final_text:
                full_text, last_text = merge_stream_text(full_text, last_text, final_text)
    except Exception as e:
        print(f"\n收尾时调用 is_final=True 失败：{e}", file=sys.stderr)

    # 程序结束前，如果还有残留的一句，也输出一下
    if full_text:
        print("\n最终一句：", full_text)

    print("程序退出。")


if __name__ == "__main__":
    main()
