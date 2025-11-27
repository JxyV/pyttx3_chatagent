#!/usr/bin/env python3
"""
WebSocket服务器 - 支持流式输出的RAG聊天服务
"""
import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag.chain import build_chain, load_retriever, answer_question
from app.config import get_rag_config
import wave
import io
from app.services.voice import VoiceInterface, GummyRealtimeSTT
# from app.services.voice import Qwen3TTSRealtime  # 【已注释保留】DashScope API TTS


# 配置日志
logging.basicConfig(level=logging.INFO)

def extract_pcm_from_wav(wav_data: bytes) -> bytes:
    """从WAV文件中提取PCM数据"""
    try:
        # 使用BytesIO包装WAV数据
        wav_buffer = io.BytesIO(wav_data)
        
        # 使用wave模块读取WAV文件
        with wave.open(wav_buffer, 'rb') as wav_file:
            # 获取音频参数
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            logger.info(f"WAV文件信息: 采样率={sample_rate}Hz, 声道={channels}, 位深={sample_width*8}位")
            
            # 读取所有音频帧
            pcm_data = wav_file.readframes(wav_file.getnframes())
            
            logger.info(f"提取的PCM数据大小: {len(pcm_data)} 字节")
            return pcm_data
            
    except Exception as e:
        logger.error(f"提取PCM数据失败: {e}")
        return wav_data  # 如果失败，返回原始数据
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="博物馆RAG WebSocket服务", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
rag_chain = None
retriever = None
voice_interface = None
active_connections: Dict[str, WebSocket] = {}
user_sessions: Dict[str, Dict] = {}
# 会话状态管理：用于支持中断机制
session_states: Dict[str, Dict] = {}  # session_id -> {current_request_id, is_interrupted, tts_request_id}

# 从环境变量读取API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ DASHSCOPE_API_KEY环境变量未设置，请在系统环境变量中配置")
    logger.warning("Windows: setx DASHSCOPE_API_KEY \"your_api_key\"")
    logger.warning("Linux/Mac: export DASHSCOPE_API_KEY=\"your_api_key\"")
else:
    logger.info(f"✅ DASHSCOPE_API_KEY已加载: {DASHSCOPE_API_KEY[:10]}...")


class ChatMessage(BaseModel):
    question: str
    chat_history: Optional[List[Dict]] = []
    session_id: Optional[str] = None


class VoiceTranscribeRequest(BaseModel):
    audio_data: str
    session_id: Optional[str] = None


def initialize_rag_system():
    """初始化RAG系统"""
    global rag_chain, retriever, voice_interface
    try:
        logger.info("正在初始化RAG系统...")
        rag_chain, retriever = build_chain()
        logger.info("RAG系统初始化成功")
        
        # 初始化语音系统
        logger.info("正在初始化语音系统...")
        voice_interface = create_voice_interface()
        logger.info("语音系统初始化成功")
        
        return True
    except Exception as e:
        logger.error(f"RAG系统初始化失败: {e}")
        return False


def create_voice_interface() -> VoiceInterface:
    """创建语音接口 - 使用Gummy实时语音识别"""
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("DASHSCOPE_API_KEY环境变量未设置，无法初始化语音接口")
            return None
        
        # 使用GummyRealtimeSTT和Pyttsx3TTS（本地TTS）
        # GummyRealtimeSTT现在内部使用stt_client实现
        stt = GummyRealtimeSTT(api_key=api_key, model="gummy-realtime-v1")
        # tts = Qwen3TTSRealtime(api_key=api_key)  # 【已注释保留】DashScope API TTS
        voice = os.getenv("TTS_VOICE", "Cherry")
        
        # VoiceInterface 现在默认使用 Pyttsx3TTS（本地TTS）
        voice_interface = VoiceInterface(stt_model=stt, tts_model=None, voice=voice)
        
        # 检查实际创建的STT类型
        stt_type = type(voice_interface.stt).__name__
        logger.info(f"✅ 语音接口初始化成功")
        logger.info(f"   实际STT类型: {stt_type}")
        logger.info(f"   STT模型: {getattr(voice_interface.stt, 'model', 'unknown')}")
        
        return voice_interface
    except Exception as e:
        logger.error(f"语音接口创建失败: {e}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return None


async def stream_response(question: str, chat_history: List[Dict], websocket: WebSocket, 
                         auto_tts: bool = False, request_id: Optional[str] = None, 
                         session_id: Optional[str] = None):
    """流式响应生成 - 使用与multimodal_rag.py相同的流程"""
    try:
        # 发送开始信号（包含request_id）
        await websocket.send_text(json.dumps({
            "type": "response_start",
            "message": "开始生成回答...",
            "requestId": request_id
        }))
        
        if not rag_chain or not retriever:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "RAG系统未初始化"
            }))
            return
        
        # 第一步：向量检索
        logger.info(f"正在检索相关知识: {question}")
        docs = retriever.invoke(question)
        
        if not docs:
            await websocket.send_text(json.dumps({
                "type": "response_end",
                "fullResponse": "抱歉，我不确定，可能未在知识库中找到相关内容。",
                "timestamp": datetime.now().isoformat(),
                "sources": []
            }))
            return
        
        logger.info(f"检索到 {len(docs)} 个相关文档片段")
        
        # 第二步：LLM流式生成回答
        import time
        start_time = time.time()
        logger.info("正在生成回答...")
        response_text = ""
        first_chunk = True
        chunk_count = 0
        
        # 使用流式输出
        for chunk in rag_chain.stream({
            "question": question, 
            "chat_history": format_chat_history(chat_history)
        }):
            # 检查是否被中断（新请求到达）
            if session_id and session_states.get(session_id):
                if session_states[session_id]["is_interrupted"] or \
                   (request_id and session_states[session_id]["current_request_id"] != request_id):
                    logger.info(f"[{session_id}] 检测到中断，停止发送后续chunk（但文字已生成）")
                    break  # 停止发送，但文字继续生成
            
            if chunk:
                response_text += chunk
                chunk_count += 1
                await websocket.send_text(json.dumps({
                    "type": "response_chunk",
                    "content": chunk,
                    "isFirst": first_chunk,
                    "requestId": request_id
                }))
                first_chunk = False
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 回答生成完成: 耗时 {elapsed_time:.2f}秒, 共 {chunk_count} 个chunk, 总长度 {len(response_text)} 字符")
        
        # 准备引用信息
        sources = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            chunk_id = d.metadata.get("chunk_id")
            locator = f"page {page}" if page is not None else f"chunk {chunk_id}"
            sources.append({"source": source, "locator": locator})
        
        # 检查是否被中断，如果被中断则不发送TTS
        should_play_tts = auto_tts
        if session_id and session_states.get(session_id):
            if session_states[session_id]["is_interrupted"] or \
               (request_id and session_states[session_id]["current_request_id"] != request_id):
                logger.info(f"[{session_id}] 响应被中断，不播放TTS")
                should_play_tts = False
        
        # 发送结束信号（包含request_id）
        await websocket.send_text(json.dumps({
            "type": "response_end",
            "fullResponse": response_text,
            "timestamp": datetime.now().isoformat(),
            "sources": sources,
            "autoTTS": should_play_tts,  # 告诉前端是否需要自动播放语音
            "requestId": request_id
        }))
        
    except Exception as e:
        logger.error(f"流式响应生成错误: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"生成回答时发生错误: {str(e)}"
        }))


async def transcribe_audio_with_gummy(audio_data: str) -> str:
    """使用Gummy进行语音转文字"""
    try:
        if not voice_interface:
            return "语音系统未初始化"
        
        # 将base64音频数据转换为字节
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        # 使用Gummy进行转录
        transcribed_text = voice_interface.stt.transcribe(audio_bytes)
        
        return transcribed_text if transcribed_text else "未识别到语音内容"
        
    except Exception as e:
        logger.error(f"Gummy转录错误: {e}")
        return f"语音识别失败: {str(e)}"


def format_chat_history(chat_history: List[Dict]) -> str:
    """格式化聊天历史"""
    if not chat_history:
        return ""
    
    formatted = []
    for msg in chat_history[-10:]:  # 只保留最近10条消息
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'user':
            formatted.append(f"用户: {content}")
        else:
            formatted.append(f"助手: {content}")
    
    return "\n".join(formatted)


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动WebSocket服务器...")
    if not initialize_rag_system():
        logger.error("RAG系统初始化失败，服务器可能无法正常工作")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    user_sessions[session_id] = {
        "chat_history": [],
        "created_at": datetime.now()
    }
    # 初始化会话状态
    session_states[session_id] = {
        "current_request_id": None,
        "is_interrupted": False,
        "tts_request_id": None
    }
    
    logger.info(f"新WebSocket连接: {session_id}")
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "send_message":
                # 兼容前端发送的消息格式
                question = message.get("question", "") or message.get("message", "")
                request_id = message.get("requestId")  # 获取请求ID
                chat_history = user_sessions[session_id]["chat_history"]
                
                # 如果有新的请求，取消旧的请求的TTS
                if request_id and session_states[session_id]["current_request_id"]:
                    old_request_id = session_states[session_id]["current_request_id"]
                    if old_request_id != request_id:
                        logger.info(f"[{session_id}] 检测到新请求 {request_id}，取消旧请求 {old_request_id} 的TTS")
                        session_states[session_id]["is_interrupted"] = True
                        session_states[session_id]["tts_request_id"] = None
                
                # 更新会话状态
                session_states[session_id]["current_request_id"] = request_id
                session_states[session_id]["is_interrupted"] = False
                
                # 添加用户消息到历史记录
                user_sessions[session_id]["chat_history"].append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 生成流式响应（传递request_id和session_id用于中断检测）
                await stream_response(question, chat_history, websocket, auto_tts=True, 
                                    request_id=request_id, session_id=session_id)
                
            elif message["type"] == "send_audio":
                # 处理语音消息 - 使用Paraformer
                audio_data = message.get("audio_data", "")
                logger.info(f"收到语音消息: {len(audio_data)} 字节")
                
                if voice_interface:
                    try:
                        # 使用Gummy进行语音转文字
                        transcribed_text = await transcribe_audio_with_gummy(audio_data)
                        
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": transcribed_text
                        }))
                        
                        # 如果识别成功，自动处理问题
                        if transcribed_text and transcribed_text.strip():
                            # 添加用户消息到历史记录
                            user_sessions[session_id]["chat_history"].append({
                                "role": "user",
                                "content": transcribed_text,
                                "timestamp": datetime.now().isoformat(),
                                "isVoice": True
                            })
                            
                            # 生成流式响应（语音输入，自动播放语音）
                            await stream_response(transcribed_text, user_sessions[session_id]["chat_history"], websocket, auto_tts=True)
                            
                    except Exception as e:
                        logger.error(f"语音转文字失败: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"语音识别失败: {str(e)}"
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "语音系统未初始化"
                    }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
    finally:
        # 清理连接
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in user_sessions:
            del user_sessions[session_id]
        if session_id in session_states:
            del session_states[session_id]


@app.websocket("/ws/realtime-speech")
async def realtime_speech_endpoint(websocket: WebSocket):
    """实时语音识别WebSocket端点 - 使用Gummy流式识别"""
    await websocket.accept()
    client_id = f"realtime_speech_{id(websocket)}"
    logger.info(f"[{client_id}] 实时语音识别客户端连接")
    
    # 每个客户端维护自己的流式识别器状态
    stt_instance = None
    streaming_started = False

    try:
        while True:
            # 接收文本消息（JSON格式）
            message = await websocket.receive_text()
            logger.info(f"[{client_id}] 收到消息: {len(message)} 字符")
            
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'start':
                    # 开始流式识别
                    if not voice_interface or not voice_interface.stt:
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'error': '语音识别服务未初始化'
                        }))
                        continue
                    
                    if not isinstance(voice_interface.stt, GummyRealtimeSTT):
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'error': '需要GummyRealtimeSTT进行流式识别'
                        }))
                        continue
                    
                    # 创建新的STT实例用于此客户端
                    api_key = os.getenv("DASHSCOPE_API_KEY")
                    stt_instance = GummyRealtimeSTT(api_key=api_key, model="gummy-realtime-v1")
                    
                    # 定义结果回调（同步函数，使用队列传递结果）
                    result_queue = asyncio.Queue()
                    
                    # 保存主事件循环引用，用于在STT回调线程中发送消息
                    main_loop = asyncio.get_event_loop()
                    
                    def result_callback(text: str, is_final: bool):
                        """STT识别结果回调"""
                        try:
                            if not text or not text.strip():
                                return  # 忽略空文本
                            
                            # 将结果放入队列，由主循环处理
                            # 即使识别器状态变化，也要处理已经收到的识别结果
                            # 使用保存的主事件循环引用
                            asyncio.run_coroutine_threadsafe(
                                result_queue.put({
                                    'type': 'interim' if not is_final else 'final',
                                    'text': text.strip()
                                }),
                                main_loop
                            )
                            logger.info(f"[{client_id}] 识别结果已放入队列: {text[:50]}... (final: {is_final})")
                        except Exception as e:
                            logger.error(f"[{client_id}] 发送识别结果到队列失败: {e}", exc_info=True)
                    
                    # 启动流式识别
                    stt_instance.start_streaming(result_callback=result_callback)
                    streaming_started = True
                    logger.info(f"[{client_id}] 流式识别已启动")
                    
                    await websocket.send_text(json.dumps({
                        'type': 'status',
                        'message': '流式识别已启动'
                    }))
                    
                    # 启动任务处理识别结果队列
                    async def process_results():
                        logger.info(f"[{client_id}] 开始处理识别结果队列")
                        # 使用一个标志来控制循环，即使streaming_started变为False，也要处理完队列中的结果
                        should_continue = True
                        while should_continue:
                            try:
                                # 增加超时时间，确保即使streaming_started变为False，也能处理完队列中的结果
                                result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
                                logger.info(f"[{client_id}] 从队列获取识别结果: type={result.get('type')}, text={result.get('text', '')[:50]}")
                                
                                # 发送识别结果到前端（使用安全的发送方法）
                                try:
                                    await websocket.send_text(json.dumps(result))
                                    logger.info(f"[{client_id}] ✅ 识别结果已发送到前端: {result.get('text', '')[:50]}")
                                except Exception as send_error:
                                    logger.warning(f"[{client_id}] 发送识别结果到前端失败（可能WebSocket已关闭）: {send_error}")
                                    # 如果WebSocket已关闭，停止处理
                                    should_continue = False
                                    break
                                    
                            except asyncio.TimeoutError:
                                # 超时时检查是否应该继续
                                if not streaming_started:
                                    # 再等待一小段时间，确保所有结果都已放入队列
                                    await asyncio.sleep(0.5)
                                    # 尝试再获取一次结果
                                    try:
                                        result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                                        logger.info(f"[{client_id}] 延迟获取到识别结果: {result.get('text', '')[:50]}")
                                        await websocket.send_text(json.dumps(result))
                                        logger.info(f"[{client_id}] ✅ 延迟发送识别结果到前端")
                                    except (asyncio.TimeoutError, Exception):
                                        # 没有更多结果了，退出循环
                                        should_continue = False
                                        break
                                continue
                            except Exception as e:
                                logger.error(f"[{client_id}] 处理识别结果失败: {e}", exc_info=True)
                                # 检查是否是WebSocket关闭错误
                                error_str = str(e).lower()
                                if "websocket" in error_str or "connection" in error_str or "close" in error_str:
                                    should_continue = False
                                    break
                        logger.info(f"[{client_id}] 识别结果处理任务结束")
                    
                    result_task = asyncio.create_task(process_results())
                    # 保存任务引用，避免被垃圾回收
                    if not hasattr(websocket, '_result_tasks'):
                        websocket._result_tasks = []
                    websocket._result_tasks.append(result_task)
                
                elif message_type == 'audio':
                    # 处理音频数据（PCM格式）
                    audio_base64 = data.get('audio', '')
                    if audio_base64:
                        import base64
                        audio_bytes = base64.b64decode(audio_base64)
                        logger.info(f"[{client_id}] 收到音频数据: {len(audio_bytes)} 字节")
                        
                        # 检查是否是WAV格式（兼容旧版本）
                        if len(audio_bytes) > 4 and audio_bytes[:4] == b'RIFF':
                            # 提取PCM数据
                            audio_bytes = extract_pcm_from_wav(audio_bytes)
                            logger.info(f"[{client_id}] WAV转换为PCM后: {len(audio_bytes)} 字节")
                        else:
                            # 已经是PCM格式，直接使用
                            logger.debug(f"[{client_id}] 收到PCM格式音频: {len(audio_bytes)} 字节")
                        
                        if streaming_started and stt_instance:
                            # 发送音频帧到流式识别器
                            try:
                                success = stt_instance.send_audio_frame(audio_bytes)
                                if success:
                                    logger.info(f"[{client_id}] ✅ 音频帧已发送到识别器: {len(audio_bytes)} 字节")
                                else:
                                    # send_audio_frame返回False表示识别器已停止
                                    logger.warning(f"[{client_id}] ⚠️ 识别器已停止，无法发送音频帧")
                                    streaming_started = False
                                    stt_instance = None
                                    # 通知前端识别器已停止
                                    await websocket.send_text(json.dumps({
                                        'type': 'error',
                                        'error': '语音识别器已停止，请重新开始录音'
                                    }))
                            except (ConnectionError, OSError) as e:
                                # WebSocket连接中断，需要重新建立连接
                                logger.warning(f"[{client_id}] ⚠️ 发送音频帧时连接中断: {e}")
                                streaming_started = False
                                try:
                                    stt_instance.stop_streaming()
                                except:
                                    pass
                                stt_instance = None
                                # 通知前端连接中断
                                await websocket.send_text(json.dumps({
                                    'type': 'error',
                                    'error': '语音识别连接中断，请重新开始录音'
                                }))
                            except Exception as e:
                                logger.error(f"[{client_id}] ❌ 发送音频帧失败: {e}")
                                # 检查是否是识别器停止的错误
                                error_str = str(e).lower()
                                if "stopped" in error_str or "has stopped" in error_str:
                                    logger.warning(f"[{client_id}] ⚠️ 识别器已停止，重置状态")
                                    streaming_started = False
                                    stt_instance = None
                                    await websocket.send_text(json.dumps({
                                        'type': 'error',
                                        'error': '语音识别器已停止，请重新开始录音'
                                    }))
                        else:
                            logger.warning(f"[{client_id}] ⚠️ 流式识别未启动或STT实例不存在 (streaming_started={streaming_started}, stt_instance={stt_instance is not None})")
                            # 如果没有启动流式识别，使用同步识别
                            if voice_interface and voice_interface.stt:
                                text = voice_interface.stt.transcribe(audio_bytes)
                                if text and text.strip():
                                    await websocket.send_text(json.dumps({
                                        'type': 'final',
                                        'text': text
                                    }))
                        
                elif message_type == 'end':
                    # 结束流式识别
                    if streaming_started and stt_instance:
                        try:
                            final_text = stt_instance.stop_streaming()
                            streaming_started = False
                            logger.info(f"[{client_id}] 流式识别已停止")
                            
                            # 如果有最终识别结果，发送给前端
                            if final_text and final_text.strip():
                                try:
                                    await websocket.send_text(json.dumps({
                                        'type': 'final',
                                        'text': final_text
                                    }))
                                    logger.info(f"[{client_id}] ✅ 最终识别结果已发送: {final_text}")
                                except Exception as send_error:
                                    logger.warning(f"[{client_id}] 发送最终识别结果失败（WebSocket可能已关闭）: {send_error}")
                        except Exception as e:
                            logger.warning(f"[{client_id}] 停止流式识别时出错（可能连接已关闭）: {e}")
                            streaming_started = False
                            # 即使出错，也尝试获取已收集的结果
                            if stt_instance and hasattr(stt_instance, 'callback'):
                                callback = stt_instance.callback
                                if callback and hasattr(callback, 'sentences') and callback.sentences:
                                    final_text = " ".join(callback.sentences).strip()
                                    if final_text:
                                        try:
                                            await websocket.send_text(json.dumps({
                                                'type': 'final',
                                                'text': final_text
                                            }))
                                            logger.info(f"[{client_id}] ✅ 从错误中恢复最终识别结果: {final_text}")
                                        except Exception:
                                            pass
                            stt_instance = None
                    
                    try:
                        await websocket.send_text(json.dumps({
                            'type': 'status',
                            'message': '语音识别已结束'
                        }))
                    except Exception as e:
                        logger.debug(f"[{client_id}] 发送状态消息失败（WebSocket可能已关闭）: {e}")
                    
                else:
                    logger.warning(f"[{client_id}] 未知消息类型: {message_type}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"[{client_id}] JSON解析失败: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'error': f'消息格式错误: {str(e)}'
                }))
            except Exception as e:
                logger.error(f"[{client_id}] 处理消息失败: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'error': f'处理消息失败: {str(e)}'
                }))

    except WebSocketDisconnect:
        logger.info(f"[{client_id}] 实时语音识别客户端断开")
        if streaming_started and stt_instance:
            try:
                stt_instance.stop_streaming()
            except Exception as e:
                logger.debug(f"[{client_id}] 清理连接时出错（可忽略）: {e}")
    except Exception as e:
        logger.error(f"[{client_id}] 实时语音识别错误: {e}", exc_info=True)
        if streaming_started and stt_instance:
            try:
                stt_instance.stop_streaming()
            except Exception as e:
                logger.debug(f"[{client_id}] 清理连接时出错（可忽略）: {e}")
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'error': str(e)
            }))
        except:
            pass
    finally:
        # 确保资源清理
        if stt_instance:
            try:
                stt_instance.stop_streaming()
            except:
                pass
            stt_instance = None


@app.websocket("/ws/tts")
async def tts_endpoint(websocket: WebSocket):
    """单次TTS WebSocket端点 - 使用pyttsx3本地TTS"""
    await websocket.accept()
    client_id = f"tts_single_{id(websocket)}"
    logger.info(f"[{client_id}] TTS客户端连接")
    
    try:
        import base64
        
        # 接收文本
        data = await websocket.receive_text()
        logger.info(f"[{client_id}] 收到前端消息: {len(data)} 字符")
        message = json.loads(data)
        logger.info(f"[{client_id}] 解析后的消息类型: {message.keys()}")
        
        if message.get('text'):
            text = message.get('text')
            logger.info(f"[{client_id}] 收到TTS请求，文本长度: {len(text)} 字符")
            logger.debug(f"[{client_id}] 文本内容: {text[:100]}...")
            
            try:
                # 使用pyttsx3本地TTS进行语音合成
                logger.info(f"[{client_id}] 开始使用pyttsx3本地TTS合成，文本: {text}")
                
                # 使用全局voice_interface的TTS模型（避免重复初始化pyttsx3引擎）
                # 在后台线程中执行TTS合成，避免阻塞WebSocket
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                def synthesize_audio():
                    if not voice_interface:
                        logger.warning(f"[{client_id}] 全局voice_interface未初始化，创建临时实例")
                        from app.services.voice import VoiceInterface
                        temp_voice = VoiceInterface()
                        return temp_voice.tts.synthesize(text)
                    else:
                        return voice_interface.tts.synthesize(text)
                
                # 在线程池中执行TTS合成（避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    audio_data = await loop.run_in_executor(executor, synthesize_audio)
                
                if audio_data:
                    # 将音频数据编码为base64
                    wav_base64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    logger.info(f"[{client_id}] WAV数据大小: {len(audio_data)} 字节")
                    
                    # 发送完整的WAV音频
                    await websocket.send_text(json.dumps({
                        'type': 'audio',
                        'audio': wav_base64
                    }))
                    
                    logger.info(f"[{client_id}] ✅ 完整音频已发送到前端")
                else:
                    logger.warning(f"[{client_id}] ⚠️ 没有生成音频数据")
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'error': '未生成音频数据'
                    }))
                
                # 发送结束信号
                logger.info(f"[{client_id}] 发送结束信号到前端")
                await websocket.send_text(json.dumps({'type': 'end'}))
                logger.info(f"[{client_id}] 结束信号已发送")
                
            except Exception as e:
                logger.error(f"[{client_id}] TTS合成异常: {e}", exc_info=True)
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'error': f'TTS合成失败: {str(e)}'
                }))
            
    except WebSocketDisconnect:
        logger.info(f"[{client_id}] TTS客户端断开")
    except Exception as e:
        logger.error(f"[{client_id}] TTS错误: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'error': f'TTS错误: {str(e)}'
            }))
        except:
            pass


# ============================================================================
# 【已注释保留】DashScope API TTS实现 - qwen3-tts-flash-realtime流式API
# ============================================================================
# @app.websocket("/ws/tts")
# async def tts_endpoint(websocket: WebSocket):
#     """单次TTS WebSocket端点 - 使用qwen3-tts-flash-realtime流式API"""
#     await websocket.accept()
#     client_id = f"tts_single_{id(websocket)}"
#     logger.info(f"[{client_id}] TTS客户端连接")
#     
#     try:
#         import base64
#         import websocket as ws_client
#         import threading
#         import io
#         import wave
#         
#         # 检查API Key
#         api_key = os.getenv("DASHSCOPE_API_KEY")
#         if not api_key:
#             await websocket.send_text(json.dumps({
#                 'type': 'error',
#                 'error': 'DASHSCOPE_API_KEY未设置'
#             }))
#             return
#         
#         # 接收文本
#         data = await websocket.receive_text()
#         logger.info(f"[{client_id}] 收到前端消息: {len(data)} 字符")
#         message = json.loads(data)
#         logger.info(f"[{client_id}] 解析后的消息类型: {message.keys()}")
#         
#         if message.get('text'):
#             text = message.get('text')
#             logger.info(f"[{client_id}] 收到TTS请求，文本长度: {len(text)} 字符")
#             logger.debug(f"[{client_id}] 文本内容: {text[:100]}...")
#             
#             try:
#                 # 使用官方WebSocket API进行TTS合成
#                 audio_chunks = []
#                 synthesis_error = None
#                 synthesis_complete = threading.Event()
#                 
#                 # 保存主事件循环引用，用于在后台线程中发送消息
#                 main_loop = asyncio.get_event_loop()
#                 
#                 logger.info(f"[{client_id}] 开始使用WebSocket API进行TTS合成，文本: {text}")
#                 
#                 # 使用WebSocket连接qwen3-tts-flash-realtime
#                 logger.info(f"[{client_id}] 使用WebSocket连接qwen3-tts-flash-realtime进行TTS合成")
#                 
#                 # WebSocket URL和请求头
#                 api_url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"
#                 headers = [f"Authorization: Bearer {api_key}"]
#                 
#                 def on_open(ws):
#                     logger.info(f"[{client_id}] TTS WebSocket连接已打开")
#                     
#                     # 1. 发送session.update事件初始化会话
#                     session_update = {
#                         "type": "session.update",
#                         "session": {
#                             "mode": "server_commit",
#                             "voice": "Cherry"
#                         }
#                     }
#                     ws.send(json.dumps(session_update))
#                     logger.info(f"[{client_id}] 已发送session.update")
#                     
#                     # 2. 发送input_text_buffer.append事件添加文本
#                     text_message = {
#                         "type": "input_text_buffer.append",
#                         "text": text
#                     }
#                     ws.send(json.dumps(text_message))
#                     logger.info(f"[{client_id}] 已发送文本: {text}")
#                     
#                     # 3. 发送input_text_buffer.commit事件开始合成
#                     commit_message = {
#                         "type": "input_text_buffer.commit"
#                     }
#                     ws.send(json.dumps(commit_message))
#                     logger.info(f"[{client_id}] 已发送commit信号")
#                     
#                     # 4. 发送session.finish事件结束会话
#                     import time
#                     time.sleep(0.5)
#                     finish_message = {
#                         "type": "session.finish"
#                     }
#                     ws.send(json.dumps(finish_message))
#                     logger.info(f"[{client_id}] 已发送session.finish")
#                 
#                 def on_message(ws, message):
#                     try:
#                         data = json.loads(message)
#                         logger.info(f"[{client_id}] 收到响应: {data.get('type', 'unknown')}")
#                         
#                         if data.get('type') == 'session.created':
#                             session_info = data.get('session', {})
#                             logger.info(f"[{client_id}] 会话已创建: {session_info.get('id')}, 模式: {session_info.get('mode')}, 音色: {session_info.get('voice')}")
#                         
#                         elif data.get('type') == 'session.updated':
#                             logger.info(f"[{client_id}] 会话已更新")
#                         
#                         elif data.get('type') == 'input_text_buffer.committed':
#                             logger.info(f"[{client_id}] 文本已提交到服务端")
#                         
#                         elif data.get('type') == 'response.created':
#                             logger.info(f"[{client_id}] 服务端开始生成响应")
#                         
#                         elif data.get('type') == 'response.output_item.added':
#                             logger.info(f"[{client_id}] 新的输出内容")
#                         
#                         elif data.get('type') == 'response.content_part.added':
#                             logger.info(f"[{client_id}] 新的内容部分")
#                         
#                         elif data.get('type') == 'response.audio.delta':
#                             # 接收音频数据，实时转发给前端，并保存用于最终合并
#                             audio_data = data.get('delta', '')
#                             if audio_data:
#                                 logger.debug(f"[{client_id}] 收到音频数据: {len(audio_data)} 字符")
#                                 audio_chunks.append(audio_data)
#                                 
#                                 # 将PCM数据转换为带WAV头的片段，实时发送到前端
#                                 try:
#                                     pcm_bytes = base64.b64decode(audio_data)
#                                     wav_buffer = io.BytesIO()
#                                     with wave.open(wav_buffer, 'wb') as wf:
#                                         wf.setnchannels(1)
#                                         wf.setsampwidth(2)
#                                         wf.setframerate(24000)
#                                         wf.writeframes(pcm_bytes)
#                                     wav_chunk_base64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')
#                                     
#                                     asyncio.run_coroutine_threadsafe(
#                                         websocket.send_text(json.dumps({
#                                             'type': 'audio_chunk',
#                                             'audio': wav_chunk_base64
#                                         })),
#                                         main_loop
#                                     )
#                                 except Exception as chunk_error:
#                                     logger.error(f"[{client_id}] 实时发送音频片段失败: {chunk_error}", exc_info=True)
#                         
#                         elif data.get('type') == 'response.content_part.done':
#                             logger.info(f"[{client_id}] 内容部分完成")
#                         
#                         elif data.get('type') == 'response.output_item.done':
#                             logger.info(f"[{client_id}] 输出项完成")
#                         
#                         elif data.get('type') == 'response.audio.done':
#                             logger.info(f"[{client_id}] 音频生成完成")
#                         
#                         elif data.get('type') == 'response.done':
#                             logger.info(f"[{client_id}] 响应完成")
#                             synthesis_complete.set()
#                         
#                         elif data.get('type') == 'session.finished':
#                             logger.info(f"[{client_id}] 会话结束")
#                             synthesis_complete.set()
#                         
#                         elif data.get('type') == 'error':
#                             error_msg = data.get('error', {}).get('message', '未知错误')
#                             logger.error(f"[{client_id}] TTS API错误: {error_msg}")
#                             synthesis_error = error_msg
#                             synthesis_complete.set()
#                         
#                         else:
#                             logger.warning(f"[{client_id}] 未知响应类型: {data.get('type')}")
#                             
#                     except Exception as e:
#                         logger.error(f"[{client_id}] 处理TTS响应失败: {e}")
#                         import traceback
#                         logger.error(traceback.format_exc())
#                         synthesis_error = str(e)
#                         synthesis_complete.set()
#                 
#                 def on_error(ws, error):
#                     logger.error(f"[{client_id}] TTS WebSocket错误: {error}")
#                     synthesis_error = str(error)
#                     synthesis_complete.set()
#                 
#                 def on_close(ws, close_status_code, close_msg):
#                     logger.info(f"[{client_id}] TTS WebSocket连接关闭")
#                     synthesis_complete.set()
#                 
#                 # 创建WebSocket连接
#                 ws_app = ws_client.WebSocketApp(
#                     api_url,
#                     header=headers,
#                     on_open=on_open,
#                     on_message=on_message,
#                     on_error=on_error,
#                     on_close=on_close
#                 )
#                 
#                 # 在新线程中运行WebSocket
#                 ws_thread = threading.Thread(target=ws_app.run_forever)
#                 ws_thread.daemon = True
#                 ws_thread.start()
#                 
#                 # 等待合成完成
#                 logger.info(f"[{client_id}] 等待合成完成...")
#                 completed = synthesis_complete.wait(timeout=30)
#                 
#                 if not completed:
#                     logger.warning(f"[{client_id}] TTS合成超时（30秒）")
#                 
#                 # 关闭WebSocket连接
#                 ws_app.close()
#                 logger.info(f"[{client_id}] TTS API WebSocket已关闭")
#                 
#                 # 检查是否有错误
#                 if synthesis_error:
#                     logger.error(f"[{client_id}] TTS合成失败: {synthesis_error}")
#                     await websocket.send_text(json.dumps({
#                         'type': 'error',
#                         'error': f'TTS合成失败: {synthesis_error}'
#                     }))
#                 elif audio_chunks:
#                     # 合并所有音频数据并转换为WAV格式
#                     logger.info(f"[{client_id}] 合并音频数据，共 {len(audio_chunks)} 个音频块")
#                     full_audio_base64 = ''.join(audio_chunks)
#                     
#                     try:
#                         import wave
#                         import io
#                         
#                         # 解码PCM数据
#                         pcm_data = base64.b64decode(full_audio_base64)
#                         logger.info(f"[{client_id}] PCM数据大小: {len(pcm_data)} 字节")
#                         
#                         # 创建WAV文件头
#                         wav_buffer = io.BytesIO()
#                         with wave.open(wav_buffer, 'wb') as wf:
#                             wf.setnchannels(1)  # 单声道
#                             wf.setsampwidth(2)  # 16位 = 2字节
#                             wf.setframerate(24000)  # 24kHz采样率
#                             wf.writeframes(pcm_data)
#                         
#                         # 获取WAV数据并编码为base64
#                         wav_data = wav_buffer.getvalue()
#                         wav_base64 = base64.b64encode(wav_data).decode('utf-8')
#                         
#                         logger.info(f"[{client_id}] WAV数据大小: {len(wav_data)} 字节")
#                         
#                         # 发送完整的WAV音频
#                         await websocket.send_text(json.dumps({
#                             'type': 'audio',
#                             'audio': wav_base64
#                         }))
#                         
#                         logger.info(f"[{client_id}] ✅ 完整音频已发送到前端")
#                         
#                     except Exception as e:
#                         logger.error(f"[{client_id}] ❌ 处理音频数据失败: {e}", exc_info=True)
#                         await websocket.send_text(json.dumps({
#                             'type': 'error',
#                             'error': f'处理音频数据失败: {str(e)}'
#                         }))
#                 else:
#                     logger.warning(f"[{client_id}] ⚠️ 没有接收到音频数据")
#                     await websocket.send_text(json.dumps({
#                         'type': 'error',
#                         'error': '未接收到音频数据，请查看后端日志'
#                     }))
#                 
#                 # 发送结束信号
#                 logger.info(f"[{client_id}] 发送结束信号到前端")
#                 await websocket.send_text(json.dumps({'type': 'end'}))
#                 logger.info(f"[{client_id}] 结束信号已发送")
#                 
#             except Exception as e:
#                 logger.error(f"[{client_id}] TTS合成异常: {e}", exc_info=True)
#                 await websocket.send_text(json.dumps({
#                     'type': 'error',
#                     'error': f'TTS合成失败: {str(e)}'
#                 }))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatMessage):
    """HTTP流式聊天端点（用于测试）"""
    async def generate():
        try:
            if not rag_chain or not retriever:
                yield f"data: {json.dumps({'error': 'RAG系统未初始化'})}\n\n"
                return
            
            # 第一步：向量检索
            docs = retriever.invoke(request.question)
            
            if not docs:
                yield f"data: {json.dumps({'content': '抱歉，我不确定，可能未在知识库中找到相关内容。'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 第二步：LLM流式生成回答
            for chunk in rag_chain.stream({
                "question": request.question, 
                "chat_history": format_chat_history(request.chat_history)
            }):
                if chunk:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"HTTP流式响应错误: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections),
        "rag_system_ready": rag_chain is not None,
        "voice_system_ready": voice_interface is not None,
        "milvus_connected": True  # 这里可以添加实际的Milvus连接检查
    }


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
