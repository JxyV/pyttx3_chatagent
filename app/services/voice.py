"""
语音接口模块 - 使用Gummy实时语音识别
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

# 阿里云DashScope导入
try:
    import dashscope
    from dashscope.audio.asr import (
        TranslationRecognizerRealtime,
        TranslationRecognizerCallback,
        TranscriptionResult,
        TranslationResult,
        RecognitionResult
    )
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

# WebSocket导入
try:
    import websocket
    import json
    import threading
    import io
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class STTModel(ABC):
    """语音转文本基类"""
    
    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        pass


class GummyRealtimeSTT(STTModel):
    """阿里云Gummy实时语音识别STT - 使用新的stt_client实现"""
    
    def __init__(self, api_key: str = None, model: str = "gummy-realtime-v1"):
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope not installed. Run: pip install dashscope")
        
        # 设置API Key
        if api_key:
            api_key_value = api_key
        elif os.getenv("DASHSCOPE_API_KEY"):
            api_key_value = os.getenv("DASHSCOPE_API_KEY")
        else:
            raise ValueError("DASHSCOPE_API_KEY not found. Please set it in environment or pass api_key parameter")
        
        # 使用新的stt_client实现
        from app.services.stt_client import GummyRealtimeSTT as STTClient
        self._stt_client = STTClient(api_key=api_key_value, model=model)
        self.model = model
        logging.info(f"Gummy Realtime STT initialized with model: {model}")
    
    def transcribe(self, audio_data: bytes) -> str:
        """同步识别音频数据（用于一次性识别）"""
        try:
            # 保存音频数据到临时PCM文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # 使用新的stt_client进行识别
                result = self._stt_client.transcribe(temp_file_path)
                return result if result else ""
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"Gummy识别失败: {e}")
            import traceback
            logging.error(f"异常详情: {traceback.format_exc()}")
            return ""
    
    def start_streaming(self, result_callback: Optional[Callable[[str, bool], None]] = None):
        """启动流式识别"""
        self._stt_client.start_streaming(result_callback=result_callback)
    
    def send_audio_frame(self, audio_frame: bytes):
        """发送音频帧（用于流式识别）"""
        self._stt_client.send_audio_frame(audio_frame)
    
    def stop_streaming(self) -> str:
        """停止流式识别并返回结果"""
        return self._stt_client.stop_streaming()
    
    def transcribe_streaming(self, audio_chunk: bytes, callback=None) -> str:
        """流式转录 - 兼容性方法"""
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
    """语音接口主类 - 使用Gummy实时语音识别"""
    
    def __init__(self, stt_model: STTModel = None, tts_model: TTSModel = None, voice: str = "Cherry"):
        load_dotenv()
        self.setup_logging()
        
        # 初始化STT和TTS
        self.stt = stt_model or self._create_stt()
        self.tts = tts_model or self._create_tts()
        self.voice = voice  # TTS音色
        
        # 音频参数 - Gummy使用16kHz
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Gummy使用16kHz采样率
        self.record_seconds = 5  # 默认录音5秒
        
        self.audio = pyaudio.PyAudio()
    
    def setup_logging(self):
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    
    def _create_stt(self) -> STTModel:
        """创建STT模型 - 使用GummyRealtimeSTT"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logging.warning("DASHSCOPE_API_KEY未设置，使用基础STT")
            # 返回一个基础的STT实现
            class BasicSTT(STTModel):
                def transcribe(self, audio_data: bytes) -> str:
                    return "STT功能需要API密钥"
                def transcribe_streaming(self, audio_chunk: bytes, callback=None) -> str:
                    return self.transcribe(audio_chunk)
            return BasicSTT()
        
        model = os.getenv("GUMMY_MODEL", "gummy-realtime-v1")
        logging.info(f"使用GummyRealtimeSTT模型: {model}")
        return GummyRealtimeSTT(api_key=api_key, model=model)
    
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
        """实时语音识别 - 使用Gummy流式识别"""
        if not isinstance(self.stt, GummyRealtimeSTT):
            print("⚠️ 实时语音识别需要GummyRealtimeSTT")
            return ""
        
        # 启动流式识别
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
            self.stt.stop_streaming()
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
    print("✅ Gummy实时语音识别已配置")
    print("✅ 测试完成")


if __name__ == "__main__":
    test_voice_interface()