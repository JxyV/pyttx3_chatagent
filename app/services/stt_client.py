"""Gummy实时语音识别客户端"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import dashscope
from dashscope.audio.asr import (
    TranscriptionResult,
    TranslationRecognizerCallback,
    TranslationRecognizerRealtime,
)

logger = logging.getLogger(__name__)


class _GummyRecognitionCollector(TranslationRecognizerCallback):
    """收集 gummy-realtime-v1 实时识别结果"""

    def __init__(self, result_callback=None) -> None:
        self.sentences: list[str] = []
        self.interim_results: list[str] = []  # 中间结果
        self.error: Exception | None = None
        self._event = threading.Event()
        self._finished = False
        self.result_callback = result_callback  # 可选的回调函数，用于实时返回结果

    def on_open(self) -> None:
        logger.debug("Gummy识别连接已打开")

    def on_close(self) -> None:
        logger.debug("Gummy识别连接已关闭")
        self._finished = True
        self._event.set()

    def on_event(
        self,
        request_id: str,
        transcription_result: TranscriptionResult | None,
        translation_result,  # 不使用翻译，但需要接收参数
        usage,
    ) -> None:
        """接收识别结果"""
        try:
            if transcription_result is not None:
                text = transcription_result.text
                if text:
                    is_final = transcription_result.is_sentence_end
                    if is_final:
                        # 完整句子
                        self.sentences.append(text)
                        logger.info("收到完整句子: %s", text)
                        # 调用回调函数（如果提供）
                        if self.result_callback:
                            try:
                                self.result_callback(text, True)
                            except Exception as e:
                                logger.error("回调函数执行失败: %s", e)
                    else:
                        # 中间结果
                        logger.debug("收到中间结果: %s", text)
                        self.interim_results.append(text)
                        # 调用回调函数（如果提供）
                        if self.result_callback:
                            try:
                                self.result_callback(text, False)
                            except Exception as e:
                                logger.error("回调函数执行失败: %s", e)
        except Exception as exc:
            logger.error("处理识别结果时出错: %s", exc, exc_info=True)
            self.error = exc
            self._event.set()

    def on_complete(self) -> None:
        """识别完成"""
        logger.debug("Gummy识别完成")
        self._finished = True
        self._event.set()

    def on_error(self, result) -> None:
        """识别出错"""
        error_msg = str(result) if result else "未知错误"
        logger.error("Gummy识别错误: %s", error_msg)
        self.error = RuntimeError(f"语音识别失败: {error_msg}")
        self._event.set()

    def wait(self, timeout: float | None = None) -> str:
        """等待识别完成并返回结果"""
        finished = self._event.wait(timeout=timeout)
        if not finished:
            raise TimeoutError("语音识别超时")
        if self.error:
            raise self.error
        # 合并所有完整句子
        result = " ".join(self.sentences).strip()
        return result  # 允许返回空字符串，由调用方判断


class GummyRealtimeSTT:
    """封装 gummy-realtime-v1 流式识别"""

    def __init__(self, api_key: str, model: str = "gummy-realtime-v1") -> None:
        dashscope.api_key = api_key
        self.model = model
        self.recognizer = None
        self.callback = None

    def transcribe(self, audio_path: str | Path, sample_rate: int = 16000) -> str:
        """
        对本地音频文件进行流式识别
        
        Args:
            audio_path: 音频文件路径（PCM 格式）
            sample_rate: 采样率（默认 16000）
        
        Returns:
            识别结果文本
        """
        callback = _GummyRecognitionCollector()
        
        # 创建 TranslationRecognizerRealtime 实例
        recognizer = TranslationRecognizerRealtime(
            model=self.model,
            format="pcm",
            sample_rate=sample_rate,
            transcription_enabled=True,
            translation_enabled=False,
            callback=callback,
        )

        logger.info("启动 gummy-realtime-v1 流式识别，模型：%s", self.model)

        try:
            recognizer.start()
            
            # 流式发送音频数据
            for chunk in self._read_chunks(audio_path):
                recognizer.send_audio_frame(chunk)
            
            # 停止识别并等待结果
            recognizer.stop()
            
            # 等待识别完成
            text = callback.wait(timeout=60)
            logger.info("STT 识别完成：%s", text)
            return text
            
        except Exception as exc:
            logger.error("Gummy STT 失败：%s", exc, exc_info=True)
            raise RuntimeError(
                "语音识别失败，请检查 API Key、模型权限及网络连接"
            ) from exc
        finally:
            # 确保关闭连接
            close_fn = getattr(recognizer, "close", None)
            if callable(close_fn):
                close_fn()

    def start_streaming(self, result_callback=None, sample_rate: int = 16000):
        """
        启动流式识别（用于WebSocket实时识别）
        
        Args:
            result_callback: 可选的回调函数，接收 (text: str, is_final: bool) 参数
            sample_rate: 采样率（默认 16000）
        """
        if self.recognizer is not None:
            logger.warning("流式识别已在运行中")
            return
        
        self.callback = _GummyRecognitionCollector(result_callback=result_callback)
        
        self.recognizer = TranslationRecognizerRealtime(
            model=self.model,
            format="pcm",
            sample_rate=sample_rate,
            transcription_enabled=True,
            translation_enabled=False,
            callback=self.callback,
        )
        
        logger.info("启动 gummy-realtime-v1 流式识别")
        self.recognizer.start()

    def send_audio_frame(self, audio_frame: bytes):
        """
        发送音频帧（用于流式识别）
        
        Args:
            audio_frame: PCM格式的音频数据
        """
        if self.recognizer is None:
            logger.warning("流式识别未启动，无法发送音频帧")
            return
        
        try:
            self.recognizer.send_audio_frame(audio_frame)
        except Exception as e:
            logger.error(f"发送音频帧失败: {e}")

    def stop_streaming(self) -> str:
        """
        停止流式识别并返回结果
        
        Returns:
            识别结果文本
        """
        if self.recognizer is None:
            return ""
        
        try:
            self.recognizer.stop()
            # 等待识别完成
            text = self.callback.wait(timeout=30)
            return text
        except Exception as e:
            logger.error(f"停止流式识别失败: {e}")
            return ""
        finally:
            # 清理
            close_fn = getattr(self.recognizer, "close", None)
            if callable(close_fn):
                close_fn()
            self.recognizer = None
            self.callback = None

    @staticmethod
    def _read_chunks(path: str | Path, chunk_size: int = 3200):
        """
        读取音频文件并分块返回
        
        Args:
            path: 文件路径
            chunk_size: 每次读取的字节数（建议 3200，约 100ms 的音频）
        
        Yields:
            音频数据块
        """
        with open(path, "rb") as file:
            while data := file.read(chunk_size):
                yield data


