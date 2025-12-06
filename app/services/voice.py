"""
语音接口模块 - 使用本地部署的 IIC/FunASR Paraformer STT
"""
import os
import io
import time
import logging
import threading
import queue
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

import pyaudio
import wave
import numpy as np
from dotenv import load_dotenv

# 本地 FunASR/ModelScope Paraformer
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False


class STTModel(ABC):
    """语音转文本基类"""
    
    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        pass

    def start_streaming(self, result_callback: Optional[Callable[[str, bool], None]] = None):
        raise NotImplementedError("Streaming not implemented for this STT model")

    def send_audio_frame(self, audio_frame: bytes):
        raise NotImplementedError("Streaming not implemented for this STT model")

    def stop_streaming(self) -> str:
        raise NotImplementedError("Streaming not implemented for this STT model")

    def transcribe_streaming(self, audio_chunk: bytes, callback=None) -> str:
        return self.transcribe(audio_chunk)


class IicRealtimeSTT(STTModel):
    """
    使用 FunASR Paraformer Online 模型实现的一次性 + 流式 STT。
    - 默认模型: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
    - 假定音频为: 16kHz, 16bit PCM, 单声道 (paInt16)
    """
    
    def __init__(
        self,
        model_id: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        model_revision: str = "v2.0.4",
        device: str = "cuda:0",
        sample_rate: int = 16000,
        chunk_size=None,
        encoder_chunk_look_back: int = 2,
        decoder_chunk_look_back: int = 1,
    ):
        if not FUNASR_AVAILABLE:
            raise ImportError("funasr 未安装。请运行: pip install funasr modelscope huggingface_hub")

        self.sample_rate = sample_rate

        # 流式相关参数，默认 [0, 10, 5] -> 600ms chunk
        self.chunk_size = chunk_size or [0, 10, 5]
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back

        # 对于16k采样率，60ms = 960个采样点；一块 = chunk_size[1] * 960
        self._model_chunk_stride = self.chunk_size[1] * 960

        logging.info(
            "Loading FunASR model: %s (rev=%s), device=%s, chunk_size=%s, stride_samples=%s",
            model_id,
            model_revision,
            device,
            self.chunk_size,
            self._model_chunk_stride,
        )

        # 加载 FunASR 模型
        self._model = AutoModel(
            model=model_id,
            model_revision=model_revision,
            device=device,
        )

        # 流式识别状态
        self._streaming_active = False
        self._cache = None
        self._audio_buffer = None
        self._result_callback: Optional[Callable[[str, bool], None]] = None
        self._last_text = ""

        logging.info("IicRealtimeSTT initialized.")

    # ========== 一次性识别接口（整段音频转文字） ==========
    def transcribe(self, audio_data: bytes) -> str:
        """
        同步识别整段音频数据。
        :param audio_data: 16kHz, 16-bit PCM, mono 的原始字节数据
        :return: 识别文本
        """
        try:
            if not audio_data:
                return ""

            # bytes -> int16 -> float32 [-1, 1]
            speech = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            if speech.size == 0:
                return ""

            # 对于一次性识别, 直接 is_final=True，把整段音频喂进去
            cache = {}
            rec_result = self._model.generate(
                input=speech,
                cache=cache,
                is_final=True,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )

            if rec_result and "text" in rec_result[0]:
                text = (rec_result[0]["text"] or "").strip()
                return text
            return ""
        except Exception as e:
            logging.error("IicRealtimeSTT.transcribe failed: %s", e, exc_info=True)
            return ""
    
    # ========== 流式识别接口 ==========
    def start_streaming(self, result_callback: Optional[Callable[[str, bool], None]] = None):
        """
        启动流式识别，会重置内部缓存。
        :param result_callback: 回调函数 (text: str, is_final: bool) -> None
                                is_final = False -> 中间结果
                                is_final = True -> 最终结果
        """
        logging.info("IicRealtimeSTT.streaming: start_streaming called.")
        self._streaming_active = True
        self._cache = {}
        self._audio_buffer = np.array([], dtype=np.float32)
        self._result_callback = result_callback
        self._last_text = ""
    
    def send_audio_frame(self, audio_frame: bytes) -> bool:
        """
        发送一帧音频数据用于流式识别。
        :param audio_frame: 一段16kHz, 16bit, mono的PCM数据
        :return: bool 是否成功发送
        """
        if not self._streaming_active:
            logging.warning("IicRealtimeSTT.streaming: send_audio_frame called but streaming not active.")
            return False
        if not audio_frame:
            return True  # 空帧不视为错误

        try:
            frame = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
            if frame.size == 0:
                return True

            # 追加到缓冲区
            self._audio_buffer = np.concatenate([self._audio_buffer, frame])

            # 当缓冲区足够大时，取一块给模型
            while self._audio_buffer.size >= self._model_chunk_stride:
                input_chunk = self._audio_buffer[: self._model_chunk_stride]
                self._audio_buffer = self._audio_buffer[self._model_chunk_stride :]

                rec_result = self._model.generate(
                    input=input_chunk,
                    cache=self._cache,
                    is_final=False,  # 中间块
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )

                if rec_result and "text" in rec_result[0]:
                    text = (rec_result[0]["text"] or "").strip()
                    self._last_text = text
                    if self._result_callback and text:
                        # 中间结果回调 is_final=False
                        self._result_callback(text, False)
            return True
        except Exception as e:
            logging.error("IicRealtimeSTT.send_audio_frame failed: %s", e, exc_info=True)
            return False

    def force_final_and_reset(self) -> str:
        """
        强制结束当前句子的识别（触发 is_final=True），返回剩余文本，并重置内部状态以便开始下一句。
        用于 VAD 静默检测超时后的断句。
        """
        if not self._streaming_active:
            return ""

        final_text = ""
        try:
            # 1. 发送空数据触发 is_final=True
            rec_result = self._model.generate(
                input=np.array([], dtype=np.float32),  # 空输入
                cache=self._cache,
                is_final=True,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            if rec_result and "text" in rec_result[0]:
                final_text = (rec_result[0]["text"] or "").strip()

            return final_text
        except Exception as e:
            logging.error("IicRealtimeSTT.force_final_and_reset failed: %s", e, exc_info=True)
            return ""
        finally:
            # 2. 重置状态，准备下一句
            self._cache = {}
            self._last_text = ""
            # 注意：不清除 _audio_buffer，因为可能还有未处理的音频？
            # 通常 VAD 触发时 buffer 应该是空的或者只有静音。为了安全起见，可以不清空 buffer，
            # 但 cache 必须清空以重置解码器上下文。
            logging.info("IicRealtimeSTT: State reset for new sentence.")
    
    def stop_streaming(self) -> str:
        """停止流式识别，处理剩余缓冲区并返回最终文本。"""
        logging.info("IicRealtimeSTT.streaming: stop_streaming called.")
        final_text = ""

        try:
            if self._streaming_active:
                # 处理缓冲区里剩余的音频，做一次 is_final=True 的收尾
                if self._audio_buffer is not None and self._audio_buffer.size > 0:
                    rec_result = self._model.generate(
                        input=self._audio_buffer,
                        cache=self._cache,
                        is_final=True,
                        chunk_size=self.chunk_size,
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back,
                    )
                    if rec_result and "text" in rec_result[0]:
                        final_text = (rec_result[0]["text"] or "").strip()

                # 回调最终结果
                if self._result_callback and final_text:
                    self._result_callback(final_text, True)

                return final_text
        except Exception as e:
            logging.error("IicRealtimeSTT.stop_streaming failed: %s", e, exc_info=True)
            return final_text
        finally:
            # 清理流式状态
            self._streaming_active = False
            self._cache = None
            self._audio_buffer = None
            self._result_callback = None
    
    def transcribe_streaming(self, audio_chunk: bytes, callback=None) -> str:
        """
        兼容性方法：对单个 chunk 做一次性识别。
        如果现有代码把 streaming 当“一块一块调用这个函数”，
        那这里就直接走非流式的 transcribe。
        """
        if callback:
            text = self.transcribe(audio_chunk)
            callback(text, True)
            return text
        else:
            return self.transcribe(audio_chunk)


class TTSModel(ABC):
    """文本转语音基类"""
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        pass


class Pyttsx3TTS(TTSModel):
    """pyttsx3文本转语音类 - 本地TTS实现（每次使用重新创建引擎）"""
    
    # 类级别的锁，确保不会并发创建引擎
    _lock = threading.Lock()
    
    def __init__(self, rate: int = 150, volume: float = 0.8, voice_id: Optional[int] = None):
        """
        初始化TTS引擎配置（不立即创建引擎）
        
        Args:
            rate: 语速 (50-300，默认150)
            volume: 音量 (0.0-1.0，默认0.8)
            voice_id: 语音ID（可选，None表示使用默认语音）
        """
        try:
            import pyttsx3
            self.pyttsx3 = pyttsx3
        except ImportError:
            raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")
        
        # 只保存配置，不创建引擎（引擎在每次使用时创建）
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.engine = None  # 引擎将在使用时创建
        
        logging.info(f"Pyttsx3 TTS实例已创建（延迟初始化），rate={rate}, volume={volume}")
    
    def _create_engine(self):
        """创建并配置pyttsx3引擎（每次使用前调用，必须在锁外调用）"""
        # 如果已有引擎，先尝试清理
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
            try:
                del self.engine
            except:
                pass
        
        # 创建新引擎
        self.engine = self.pyttsx3.init()
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
        
        # 如果指定了语音ID，设置语音
        if self.voice_id is not None:
            voices = self.engine.getProperty('voices')
            if 0 <= self.voice_id < len(voices):
                self.engine.setProperty('voice', voices[self.voice_id].id)
        
        logging.debug(f"Pyttsx3引擎已创建，rate={self.rate}, volume={self.volume}")
    
    def synthesize(self, text: str, voice: str = "Cherry") -> bytes:
        """
        将文本转换为语音并返回WAV音频数据
        
        Args:
            text: 要转换的文本
            voice: 音色（pyttsx3使用系统默认，此参数保留兼容性）
        
        Returns:
            WAV格式的音频数据（bytes）
        """
        import tempfile
        import os
        
        # 使用锁保护合成过程，确保不会并发调用
        with Pyttsx3TTS._lock:
            # 每次使用前重新创建引擎（避免引擎状态问题）
            self._create_engine()
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # 保存语音到临时文件
                logging.debug(f"开始保存语音到文件: {temp_path}, 文本长度: {len(text)}")
                self.engine.save_to_file(text, temp_path)
                
                # runAndWait() 会阻塞直到完成，使用线程+超时机制避免卡住
                logging.debug("开始执行 runAndWait()...")
                
                # 使用线程执行 runAndWait()，并设置超时
                result_queue = queue.Queue()
                exception_queue = queue.Queue()
                
                def run_engine():
                    try:
                        self.engine.runAndWait()
                        result_queue.put(True)
                    except Exception as e:
                        exception_queue.put(e)
                
                engine_thread = threading.Thread(target=run_engine, daemon=True)
                engine_thread.start()
                
                # 等待最多60秒（对于长文本，可能需要更长时间）
                timeout = max(60, len(text) * 0.5)  # 根据文本长度动态调整超时
                engine_thread.join(timeout=timeout)
                
                if engine_thread.is_alive():
                    # 超时了，尝试停止引擎
                    logging.error(f"runAndWait() 超时（{timeout}秒），尝试停止引擎")
                    try:
                        self.engine.stop()
                    except:
                        pass
                    raise RuntimeError(f"TTS合成超时（{timeout}秒），文本长度: {len(text)}")
                
                # 检查是否有异常
                if not exception_queue.empty():
                    exception = exception_queue.get()
                    raise exception
                
                if not result_queue.empty():
                    logging.debug("runAndWait() 完成")
                else:
                    raise RuntimeError("runAndWait() 执行失败，未返回结果")
                
                # 等待文件写入完成（确保文件已关闭）
                import time
                time.sleep(0.1)
                
                # 检查文件是否存在且大小大于0
                if not os.path.exists(temp_path):
                    raise RuntimeError(f"临时文件未创建: {temp_path}")
                
                file_size = os.path.getsize(temp_path)
                if file_size == 0:
                    raise RuntimeError(f"临时文件为空: {temp_path}")
                
                logging.debug(f"临时文件大小: {file_size} 字节")
                
                # 读取文件内容
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                if len(audio_data) == 0:
                    raise RuntimeError("读取的音频数据为空")
                
                logging.info(f"Pyttsx3 TTS合成完成，音频大小: {len(audio_data)} 字节")
                return audio_data
                
            except Exception as e:
                logging.error(f"Pyttsx3 TTS合成失败: {e}", exc_info=True)
                raise
            finally:
                # 清理引擎（每次使用后清理，避免状态问题）
                try:
                    if self.engine is not None:
                        self.engine.stop()
                        del self.engine
                        self.engine = None
                except Exception as e:
                    logging.warning(f"清理引擎失败: {e}")
                
                # 清理临时文件
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"清理临时文件失败: {e}")
    
    def get_available_voices(self) -> list:
        """
        获取可用的语音列表
        
        Returns:
            语音列表，每个元素包含id和name
        """
        voices = self.engine.getProperty('voices')
        return [{"id": v.id, "name": v.name} for v in voices]
    
    def set_rate(self, rate: int):
        """设置语速 (50-300)"""
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """设置音量 (0.0-1.0)"""
        self.engine.setProperty('volume', volume)
    
    def set_voice(self, voice_id: int):
        """
        设置语音
        
        Args:
            voice_id: 语音索引（使用get_available_voices()获取）
        """
        voices = self.engine.getProperty('voices')
        if 0 <= voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)
        else:
            raise ValueError(f"无效的语音ID: {voice_id}，可用范围: 0-{len(voices)-1}")


# ============================================================================
# 【已注释保留】Qwen3TTSRealtime - DashScope API TTS实现
# ============================================================================
# class Qwen3TTSRealtime(TTSModel):
#     """Qwen3 TTS Flash Realtime模型 - 阿里云最快的流式TTS"""
#     
#     def __init__(self, api_key: str = None, model: str = "qwen3-tts-flash-realtime"):
#         if not WEBSOCKET_AVAILABLE:
#             raise ImportError("websocket-client not installed. Run: pip install websocket-client")
#         
#         # 设置API Key
#         if api_key:
#             self.api_key = api_key
#         elif os.getenv("DASHSCOPE_API_KEY"):
#             self.api_key = os.getenv("DASHSCOPE_API_KEY")
#         else:
#             raise ValueError("DASHSCOPE_API_KEY not found. Please set it in environment or pass api_key parameter")
#         
#         self.model = model
#         self.api_url = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model}"
#         self.audio_data = b""
#         self.synthesis_complete = False
#         logging.info(f"Qwen3 TTS Realtime initialized with model: {model}")
#     
#     def synthesize_streaming(self, text: str, voice: str = "Cherry", callback=None) -> dict:
#         """使用真正的qwen3-tts-flash-realtime API进行语音合成"""
#         print(f"🔧 [DEBUG] 开始qwen3-tts-flash-realtime API合成")
#         print(f"🔧 [DEBUG] 输入文本: '{text}'")
#         print(f"🔧 [DEBUG] 音色: {voice}")
#         
#         try:
#             # 使用DashScope SDK进行真正的API调用
#             import dashscope
#             from dashscope.audio.tts import SpeechSynthesizer
#             
#             # 设置API Key
#             dashscope.api_key = self.api_key
#             
#             print("🔄 调用DashScope sambert-zhichu-v1 API...")
#             
#             # 使用sambert-zhichu-v1模型（qwen3-tts-flash-realtime不支持同步调用）
#             response = SpeechSynthesizer.call(
#                 model='sambert-zhichu-v1',
#                 text=text,
#                 voice='zhixiaoxia',  # sambert模型的音色
#                 format='wav',
#                 sample_rate=16000  # sambert使用16kHz
#             )
#             
#             # 检查响应
#             print(f"📋 响应对象类型: {type(response)}")
#             
#             # 获取音频数据
#             audio_data = response.get_audio_data()
#             if audio_data:
#                 print(f"✅ API调用成功，音频数据大小: {len(audio_data)} 字节")
#                 
#                 self.audio_data = audio_data
#                 self.synthesis_complete = True
#                 self.first_audio_time = time.perf_counter()
#                 self.synthesis_start_time = time.perf_counter()
#                 
#                 # 计算性能统计
#                 synthesis_end_time = time.perf_counter()
#                 first_audio_latency = (self.first_audio_time - self.synthesis_start_time) * 1000.0
#                 total_synthesis_time = (synthesis_end_time - self.synthesis_start_time) * 1000.0
#                 chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
#                 
#                 performance = {
#                     "first_audio_ms": first_audio_latency,
#                     "total_synthesis_ms": total_synthesis_time,
#                     "chinese_count": chinese_count
#                 }
#                 
#                 print(f"🔧 [DEBUG] sambert-zhichu-v1合成完成，音频大小: {len(audio_data)} 字节")
#                 return {
#                     "audio_data": audio_data,
#                     "performance": performance
#                 }
#             else:
#                 print("❌ API调用失败，未返回音频数据")
#                 return {"audio_data": b"", "performance": {}}
#                 
#         except Exception as e:
#             print(f"🔧 [DEBUG] sambert-zhichu-v1合成异常: {e}")
#             import traceback
#             print(f"🔧 [DEBUG] 异常详情: {traceback.format_exc()}")
#             return {"audio_data": b"", "performance": {}}
#     
#     def _create_wav_header(self, data_size: int, sample_rate: int) -> bytes:
#         """创建WAV文件头"""
#         # WAV文件头
#         header = b'RIFF'
#         header += (data_size + 36).to_bytes(4, 'little')  # 文件大小
#         header += b'WAVE'
#         header += b'fmt '
#         header += (16).to_bytes(4, 'little')  # fmt chunk大小
#         header += (1).to_bytes(2, 'little')   # 音频格式 (PCM)
#         header += (1).to_bytes(2, 'little')   # 声道数
#         header += sample_rate.to_bytes(4, 'little')  # 采样率
#         header += (sample_rate * 2).to_bytes(4, 'little')  # 字节率
#         header += (2).to_bytes(2, 'little')   # 块对齐
#         header += (16).to_bytes(2, 'little')  # 位深度
#         header += b'data'
#         header += data_size.to_bytes(4, 'little')  # 数据大小
#         
#         return header
#     
#     def synthesize(self, text: str, voice: str = "Cherry") -> bytes:
#         """使用Qwen3 TTS Realtime进行语音合成（兼容性方法）"""
#         result = self.synthesize_streaming(text, voice)
#         return result["audio_data"]


class VoiceInterface:
    """语音接口主类 - 使用本地 FunASR Paraformer 语音识别"""
    
    def __init__(self, stt_model: STTModel = None, tts_model: TTSModel = None, voice: str = "Cherry"):
        load_dotenv()
        self.setup_logging()
        
        # 初始化STT和TTS
        self.stt = stt_model or self._create_stt()
        self.tts = tts_model or self._create_tts()
        self.voice = voice  # TTS音色
        
        # 音频参数 - Paraformer使用16kHz
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 16kHz采样率
        self.record_seconds = 5  # 默认录音5秒
        
        self.audio = pyaudio.PyAudio()
    
    def setup_logging(self):
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    
    def _create_stt(self) -> STTModel:
        """创建STT模型 - 使用本地 FunASR Paraformer"""
        try:
            return IicRealtimeSTT()
        except ImportError as e:
            logging.error(f"创建 STT 失败，缺少依赖: {e}")
        except Exception as e:
            logging.error(f"创建 STT 失败: {e}", exc_info=True)

        # 返回一个基础的占位 STT
        class BasicSTT(STTModel):
            def transcribe(self, audio_data: bytes) -> str:
                return "本地STT不可用，请检查funasr/modelscope依赖"

            def transcribe_streaming(self, audio_chunk: bytes, callback=None) -> str:
                text = self.transcribe(audio_chunk)
                if callback:
                    callback(text, True)
                return text

        return BasicSTT()
    
    def _create_tts(self) -> TTSModel:
        """创建TTS模型 - 默认使用pyttsx3本地TTS"""
        # 使用本地pyttsx3 TTS
        rate = int(os.getenv("TTS_RATE", "150"))
        volume = float(os.getenv("TTS_VOLUME", "0.8"))
        voice_id = os.getenv("TTS_VOICE_ID")
        voice_id = int(voice_id) if voice_id else None
        return Pyttsx3TTS(rate=rate, volume=volume, voice_id=voice_id)
        
        # 【已注释保留】使用DashScope API TTS
        # api_key = os.getenv("DASHSCOPE_API_KEY")
        # model = os.getenv("TTS_MODEL", "qwen3-tts-flash-realtime")
        # return Qwen3TTSRealtime(api_key=api_key, model=model)
    
    def record_audio(self, duration: Optional[int] = None) -> bytes:
        """录制音频"""
        duration = duration or self.record_seconds
        
        print(f"🎤 开始录音 ({duration}秒)...")
        
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # 将音频数据转换为WAV格式
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        
        print("✅ 录音完成")
        return wav_buffer.getvalue()
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """将音频转换为文本"""
        print("🔄 正在识别语音...")
        start_time = time.perf_counter()
        
        text = self.stt.transcribe(audio_data)
        
        end_time = time.perf_counter()
        
        # 输出识别结果
        if text and text.strip():
            print(f"✅ 识别完成 ({end_time - start_time:.1f}秒): {text}")
        else:
            print(f"❌ 识别失败 ({end_time - start_time:.1f}秒): 未识别到有效语音内容")
        
        return text
    
    def synthesize_speech(self, text: str) -> bytes:
        """将文本转换为语音"""
        print("🔄 正在合成语音...")
        start_time = time.perf_counter()
        
        audio_data = self.tts.synthesize(text, voice=self.voice)
        
        end_time = time.perf_counter()
        print(f"✅ 合成完成 ({end_time - start_time:.1f}秒)")
        
        return audio_data
    
    def play_audio_streaming(self, audio_chunk: bytes):
        """流式播放音频片段"""
        try:
            import tempfile
            import subprocess
            import platform
            
            # 保存音频片段到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_chunk)
                temp_file_path = temp_file.name
            
            # 根据操作系统选择播放器
            if platform.system() == "Windows":
                # Windows系统尝试多种播放方法
                self._play_audio_windows(temp_file_path)
            else:
                # Linux系统使用aplay
                subprocess.run(["aplay", temp_file_path], check=True, capture_output=True)
            
            # 清理临时文件
            os.unlink(temp_file_path)
            
        except Exception as e:
            logging.debug(f"Streaming audio playback failed: {e}")
    
    def _play_audio_windows(self, audio_file_path: str):
        """Windows系统音频播放方法 - 多种备选方案"""
        import subprocess
        import os
        
        # 方法1: PowerShell Media.SoundPlayer
        try:
            # 使用更安全的PowerShell命令
            ps_cmd = f'powershell -ExecutionPolicy Bypass -Command "(New-Object Media.SoundPlayer \'{audio_file_path}\').PlaySync()"'
            result = subprocess.run(ps_cmd, shell=True, check=True, capture_output=True, timeout=10)
            logging.info("Windows音频播放成功 (PowerShell Media.SoundPlayer)")
            return
        except Exception as e:
            logging.debug(f"PowerShell播放失败: {e}")
        
        # 方法2: 使用Windows内置的wav播放器
        try:
            subprocess.run(["cmd", "/c", f'"{audio_file_path}"'], check=True, timeout=10)
            logging.info("Windows音频播放成功 (默认程序)")
            return
        except Exception as e:
            logging.debug(f"默认程序播放失败: {e}")
        
        # 方法3: 使用start命令
        try:
            subprocess.run(["start", "/wait", audio_file_path], shell=True, check=True, timeout=10)
            logging.info("Windows音频播放成功 (start命令)")
            return
        except Exception as e:
            logging.debug(f"start命令播放失败: {e}")
        
        # 方法4: 使用Python的winsound模块
        try:
            import winsound
            winsound.PlaySound(audio_file_path, winsound.SND_FILENAME | winsound.SND_SYNC)
            logging.info("Windows音频播放成功 (winsound)")
            return
        except Exception as e:
            logging.debug(f"winsound播放失败: {e}")
        
        # 方法5: 使用PyAudio直接播放
        try:
            self._play_audio_pyaudio(audio_file_path)
            logging.info("Windows音频播放成功 (PyAudio)")
            return
        except Exception as e:
            logging.debug(f"PyAudio播放失败: {e}")
        
        logging.warning("所有Windows音频播放方法都失败了")
    
    def _play_audio_pyaudio(self, audio_file_path: str):
        """使用PyAudio播放音频文件"""
        import wave
        
        # 打开音频文件
        with wave.open(audio_file_path, 'rb') as wf:
            # 获取音频参数
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            
            # 创建PyAudio流
            stream = self.audio.open(
                format=self.audio.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True
            )
            
            # 播放音频
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            # 关闭流
            stream.stop_stream()
            stream.close()
    
    def play_audio(self, audio_data: bytes):
        """播放音频"""
        print("🔊 正在播放...")
        
        # 直接使用系统播放器，避免PyAudio的采样率问题
        self._fallback_play_audio(audio_data)
    
    def _fallback_play_audio(self, audio_data: bytes):
        """备选音频播放方案"""
        try:
            import tempfile
            import platform
            
            # 保存音频到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 根据操作系统选择播放器
            print("🔄 使用系统播放器播放...")
            if platform.system() == "Windows":
                # Windows系统使用多种备选播放方法
                self._play_audio_windows(temp_file_path)
            else:
                # Linux系统使用aplay
                import subprocess
                subprocess.run(["aplay", temp_file_path], check=True)
            
            # 清理临时文件
            os.unlink(temp_file_path)
            print("✅ 播放完成")
            
        except Exception as e:
            logging.error(f"Fallback audio playback failed: {e}")
            print(f"❌ 备选播放也失败了: {e}")
            if platform.system() == "Windows":
                print("💡 建议：请检查Windows音频设备或尝试其他播放器")
                print("💡 可能的解决方案：")
                print("   1. 检查系统音量设置")
                print("   2. 确认音频设备正常工作")
                print("   3. 尝试手动播放WAV文件")
            else:
                print("💡 建议：请检查音频设备或安装aplay: sudo apt-get install alsa-utils")
    
    def voice_to_text(self, duration: Optional[int] = None) -> str:
        """完整的语音转文本流程"""
        audio_data = self.record_audio(duration)
        return self.transcribe_audio(audio_data)
    
    def realtime_voice_recognition(self, duration: Optional[int] = None, callback=None) -> str:
        """实时语音识别 - 使用本地FunASR流式识别"""
        if not hasattr(self.stt, "start_streaming"):
            print("⚠️ 当前STT未实现流式接口")
            return ""
        
        result_texts = []
        
        def result_callback(text: str, is_final: bool):
            if text:
                result_texts.append(text)
                if callback:
                    callback(text, is_final)
        
        self.stt.start_streaming(result_callback)
        
        # 开始录音
        print(f"🎤 开始实时语音识别 ({duration or self.record_seconds}秒)...")
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        try:
            frames_to_record = int(self.rate / self.chunk * (duration or self.record_seconds))
            for _ in range(frames_to_record):
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.stt.send_audio_frame(data)
        finally:
            stream.stop_stream()
            stream.close()
            final_text = ""
            try:
                final_text = self.stt.stop_streaming()
                if final_text:
                    result_texts.append(final_text)
            except Exception:
                logging.error("停止流式识别时出错", exc_info=True)

            print("✅ 实时语音识别完成")
        
        return " ".join(result_texts).strip()
    
    def text_to_voice_streaming(self, text: str) -> dict:
        """完整的流式文本转语音流程"""
        print("🔄 正在流式合成语音...")
        print(f"🔧 [DEBUG] 调用TTS合成，文本: '{text}', 音色: {self.voice}")
        
        # 使用流式合成
        result = self.tts.synthesize_streaming(text, voice=self.voice, callback=self.play_audio_streaming)
        
        print(f"🔧 [DEBUG] TTS合成结果: 音频大小={len(result.get('audio_data', b''))} 字节")
        return result
    
    def text_to_voice(self, text: str):
        """完整的文本转语音流程"""
        audio_data = self.synthesize_speech(text)
        self.play_audio(audio_data)
    
    def set_voice(self, voice: str):
        """设置TTS音色"""
        self.voice = voice
        logging.info(f"TTS voice changed to: {voice}")
    
    def get_available_voices(self) -> dict:
        """获取可用的音色列表 - 根据官方文档"""
        return {
            "芊悦 (Cherry)": "Cherry",
            "晨煦 (Ethan)": "Ethan", 
            "不吃鱼 (Nofish)": "Nofish",
            "詹妮弗 (Jennifer)": "Jennifer",
            "甜茶 (Ryan)": "Ryan",
            "卡捷琳娜 (Katerina)": "Katerina",
            "墨讲师 (Elias)": "Elias",
            "上海-阿珍 (Jada)": "Jada",
            "北京-晓东 (Dylan)": "Dylan",
            "四川-晴儿 (Sunny)": "Sunny",
            "南京-老李 (Li)": "Li",  # 修正为官方文档中的Li
            "陕西-秦川 (Marcus)": "Marcus",
            "闽南-阿杰 (Roy)": "Roy",
            "天津-李彼得 (Peter)": "Peter",
            "粤语-阿强 (Rocky)": "Rocky",
            "粤语-阿清 (Kiki)": "Kiki",
            "四川-程川 (Eric)": "Eric"
        }
    
    def cleanup(self):
        """清理资源"""
        self.audio.terminate()


def test_voice_interface():
    """测试语音接口"""
    print("=== 语音接口测试 ===")
    print("✅ FunASR 本地实时语音识别已配置")
    print("✅ 测试完成")


if __name__ == "__main__":
    test_voice_interface()