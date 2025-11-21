"""
多模态RAG系统 - 支持语音和文本输入，完整的RAG流程
"""
import os
import sys
import time
import logging
from typing import List, Dict

from dotenv import load_dotenv
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.chain import build_chain
from app.services.voice import VoiceInterface, Qwen3TTSRealtime
from app.services.stt_client import GummyRealtimeSTT as STTClient

# 加载环境变量
load_dotenv()

# 检查API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("⚠️ 警告: DASHSCOPE_API_KEY环境变量未设置")
    print("请在系统环境变量中配置API Key:")
    print("  Windows: setx DASHSCOPE_API_KEY \"your_api_key\"")
    print("  Linux/Mac: export DASHSCOPE_API_KEY=\"your_api_key\"")
    print()
else:
    print(f"✅ DASHSCOPE_API_KEY已加载: {DASHSCOPE_API_KEY[:10]}...")

class MultimodalRAG:
    """多模态RAG系统"""
    
    def __init__(self):
        load_dotenv()
        self.setup_logging()
        
        # 初始化RAG系统
        print("🔄 正在初始化RAG系统...")
        self.chain, self.retriever = build_chain()
        print("✅ RAG系统初始化完成！")
        
        # 初始化语音系统 - 程序员可以在这里选择模型
        print("🔄 正在初始化语音系统...")
        self.voice = self._create_voice_interface()
        print("✅ 语音系统初始化完成！")
        
        # 配置
        self.auto_tts = os.getenv("AUTO_TTS", "true").lower() == "true"
        self.record_duration = int(os.getenv("RECORD_DURATION", "5"))
        self.chat_history: List[str] = []
        self.current_input_mode = "text"  # 跟踪当前输入方式
    
    def setup_logging(self):
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    
    def _create_voice_interface(self) -> VoiceInterface:
        """创建语音接口 - 使用Gummy实时语音识别"""
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置，无法初始化语音系统")
        
        # 使用新的STT客户端和Qwen3TTSRealtime
        from app.services.voice import GummyRealtimeSTT
        stt = GummyRealtimeSTT(api_key=api_key, model="gummy-realtime-v1")
        tts = Qwen3TTSRealtime(api_key=api_key)
        
        # 默认使用芊悦音色，程序员可以修改
        voice = os.getenv("TTS_VOICE", "Cherry")
        
        return VoiceInterface(stt_model=stt, tts_model=tts, voice=voice)
    
    def get_user_input(self) -> str:
        """获取用户输入（支持文本和语音）"""
        print("\n" + "="*50)
        print("选择输入方式:")
        print("1. 文本输入 (t)")
        print("2. 语音输入 (v)")  
        print("3. 退出 (q)")
        
        choice = input("请选择 (t/v/q): ").strip().lower()
        
        if choice == "q":
            return ":q"
        elif choice == "v":
            print("🎤 请说话...")
            self.current_input_mode = "voice"  # 标记为语音输入
            recognized_text = self.voice.voice_to_text(duration=self.record_duration)
            # 无论是否识别成功都返回结果，让后续逻辑处理
            return recognized_text
        else:  # 默认文本输入
            self.current_input_mode = "text"  # 标记为文本输入
            return input("你：").strip()
    
    def rag_process(self, question: str) -> Dict:
        """完整的RAG处理流程"""
        print(f"\n📝 问题: {question}")
        print("🔄 正在检索相关知识...")
        
        # 第一步：向量检索
        t_retrieval_start = time.perf_counter()
        docs = self.retriever.invoke(question)
        t_retrieval_end = time.perf_counter()
        
        if not docs:
            return {
                "answer": "抱歉，我不确定，可能未在知识库中找到相关内容。",
                "sources": [],
                "performance": {
                    "retrieval_ms": (t_retrieval_end - t_retrieval_start) * 1000.0,
                    "generation_ms": 0,
                    "total_ms": (t_retrieval_end - t_retrieval_start) * 1000.0,
                    "chinese_count": 0
                }
            }
        
        print(f"✅ 检索到 {len(docs)} 个相关文档片段")
        print("🔄 正在生成回答...")
        
        # 第二步：LLM流式生成回答
        t_generation_start = time.perf_counter()
        first_token_time = None
        answer = ""
        
        print("🤖 AI回答: ", end="", flush=True)
        
        # 使用流式输出
        for chunk in self.chain.stream({
            "question": question, 
            "chat_history": "\n".join(self.chat_history)
        }):
            if first_token_time is None:
                first_token_time = time.perf_counter()
                first_token_latency = (first_token_time - t_generation_start) * 1000.0
                print(f"\n⚡ 首token延迟: {first_token_latency:.1f}ms")
                print("🤖 AI回答: ", end="", flush=True)
            
            # 打印流式内容
            print(chunk, end="", flush=True)
            answer += chunk
        
        t_generation_end = time.perf_counter()
        print()  # 换行
        
        # 统计中文字符
        chinese_count = sum(1 for ch in answer if "\u4e00" <= ch <= "\u9fff")
        
        # 准备引用信息
        sources = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            chunk_id = d.metadata.get("chunk_id")
            locator = f"page {page}" if page is not None else f"chunk {chunk_id}"
            sources.append({"source": source, "locator": locator})
        
        # 性能统计
        first_token_latency = (first_token_time - t_generation_start) * 1000.0 if first_token_time else 0
        performance = {
            "retrieval_ms": (t_retrieval_end - t_retrieval_start) * 1000.0,
            "first_token_ms": first_token_latency,
            "generation_ms": (t_generation_end - t_generation_start) * 1000.0,
            "total_ms": (t_retrieval_end - t_retrieval_start + t_generation_end - t_generation_start) * 1000.0,
            "chinese_count": chinese_count
        }
        
        return {
            "answer": answer,
            "sources": sources,
            "performance": performance
        }
    
    def display_result(self, result: Dict):
        """显示RAG结果"""
        answer = result["answer"]
        sources = result["sources"]
        perf = result["performance"]
        
        # 流式输出已经在生成过程中显示，这里不需要重复显示
        
        # 性能统计
        print("\n--- 性能统计 ---")
        print(f"检索耗时：{perf['retrieval_ms']:.1f} ms")
        print(f"首token延迟：{perf['first_token_ms']:.1f} ms")
        print(f"生成耗时：{perf['generation_ms']:.1f} ms")
        print(f"总耗时：{perf['total_ms']:.1f} ms")
        print(f"中文字符数：{perf['chinese_count']}")
        
        # 引用信息
        if sources:
            print("\n--- 引用 ---")
            for i, s in enumerate(sources, 1):
                print(f"{i}. {s['source']} ({s['locator']})")
        
        # 根据输入方式决定是否播放语音
        if self.current_input_mode == "voice" and answer.strip():
            print("\n🔊 正在流式播放回答...")
            tts_result = self.voice.text_to_voice_streaming(answer)
            
            # 显示语音性能统计
            if tts_result and "performance" in tts_result:
                tts_perf = tts_result["performance"]
                print("\n--- 语音性能统计 ---")
                print(f"语音首token延迟：{tts_perf.get('first_audio_ms', 0):.1f} ms")
                print(f"语音总消耗时间：{tts_perf.get('total_synthesis_ms', 0):.1f} ms")
                print(f"语音中文字符数：{tts_perf.get('chinese_count', 0)}")
        elif self.current_input_mode == "text":
            print("\n💬 文本回答完成")
    
    def process_question(self, question: str):
        """处理单个问题"""
        if not question:
            print("❌ 未收到有效输入，请重试")
            return
        
        if question in {":q", "exit", "quit"}:
            return "exit"
        
        # 执行RAG流程
        result = self.rag_process(question)
        
        # 显示结果
        self.display_result(result)
        
        # 更新聊天历史
        self.chat_history.append(f"用户: {question}")
        self.chat_history.append("助手: [上一轮回答略]")
    
    def run(self):
        """运行多模态RAG系统"""
        print("🎙️ 多模态RAG对话系统")
        print("=" * 50)
        print("支持功能:")
        print("- 文本输入 → RAG检索 → 文本/语音输出")
        print("- 语音输入 → RAG检索 → 文本/语音输出")
        print("- 完整的知识库检索和引用")
        print("=" * 50)
        
        while True:
            try:
                question = self.get_user_input()
                
                if question == ":q":
                    break
                
                if not question:
                    continue
                
                result = self.process_question(question)
                if result == "exit":
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                logging.exception("发生错误: %s", e)
                print(f"❌ 发生错误: {e}")
        
        self.voice.cleanup()
        print("✅ 系统已关闭")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 命令行模式：直接处理问题
        question = " ".join(sys.argv[1:])
        rag = MultimodalRAG()
        rag.process_question(question)
        rag.voice.cleanup()
    else:
        # 交互模式
        rag = MultimodalRAG()
        rag.run()


if __name__ == "__main__":
    main()
