import os
import logging
from typing import Dict, List
from operator import itemgetter

from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def build_embeddings():
    backend = os.getenv("EMBEDDING_BACKEND", "ollama").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    if backend == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        return OllamaEmbeddings(model=model_name, base_url=base_url)
    return HuggingFaceEmbeddings(model_name=model_name)


def build_llm():
    backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # 降低温度，让输出更直接确定

    if backend == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        return ChatOpenAI(model=model, temperature=temperature, base_url=base_url)
    else:
        model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model_kwargs = {
            "num_predict": 1024,  # 增加最大生成token数，从512提升到1024
            "num_ctx": 4096,  # 增加上下文窗口，提升性能
            "keep_alive": "30m",  # 保持模型在内存中30分钟，避免重复加载
            "top_p": 0.5,  # 降低采样多样性
            "repeat_penalty": 1.1,  # 避免重复
            "num_gpu": -1,  # 使用所有可用GPU（-1表示自动检测）
            "num_thread": 4,  # 如果使用CPU，使用4个线程
        }
        logging.info(f"初始化Ollama LLM: model={model}, num_gpu={model_kwargs.get('num_gpu')}, keep_alive={model_kwargs.get('keep_alive')}")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url, model_kwargs=model_kwargs)


def load_retriever():
    from app.config import get_rag_config
    
    embeddings = build_embeddings()
    config = get_rag_config()
    milvus_config = config.get_milvus_config()
    
    # 对于Standalone版本，简化连接参数
    connection_args = {
        "host": milvus_config["host"],
        "port": milvus_config["port"]
    }
    
    vectordb = Milvus(
        collection_name=config.get_collection_name(),
        embedding_function=embeddings,
        connection_args=connection_args,
    )
    k = config.get_retrieval_config()["top_k"]
    return vectordb.as_retriever(search_kwargs={"k": k})


def format_docs_for_prompt(docs: List) -> str:
    chunks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        chunk_id = d.metadata.get("chunk_id")
        locator = f"page {page}" if page is not None else f"chunk {chunk_id}"
        chunks.append(f"[{source} | {locator}]\n{d.page_content}")
    return "\n\n---\n\n".join(chunks)


def build_chain():
    """
    Build minimal LCEL pipeline with optional chat history:
    inputs: {"question": str, "chat_history": str}
    """
    load_dotenv()
    setup_logging()

    retriever = load_retriever()
    llm = build_llm()

    system_template = (
        "【核心规则】直接输出答案，禁止任何思考过程、推理步骤、分析说明。\n\n"
        "你是一个熟悉湖北省武汉市和湖北博物馆的历史文化、美食、旅游、风土人情的知识助手，能够结合提供的文档内容和自身知识进行自然、准确的快速地回答。\n\n"
        "你的任务是：\n"
        "- 优先参考我提供的文档内容（即上下文）回答问题；\n"
        "- 如果文档信息不足或缺失，可以适当补充你自身掌握的可靠知识，回答的内容不要过少；\n"
        "- 严格按照用户要求的字数或长度进行回答，如果用户指定了字数（如'200字'、'500字'、'简短回答'等），请严格控制输出长度；\n"
        "- 禁止输出任何思考过程、解释、格式说明或AI语气的语句；\n"
        "- 直接输出自然、口语化的中文回答正文，像一个知识丰富的当地人那样娓娓道来；\n"
        "- 使用纯文本格式输出，禁止使用Markdown格式（如**粗体**、*斜体*、#标题等），不要输出星号、井号、下划线等格式符号；\n"
        "- 请务必保证措辞合理、逻辑通顺、语义信息完整。\n\n"
        "字数控制要求：\n"
        "- 如果没有明确字数要求，默认回答控制在200-300字之间；\n"
        "- 如果用户要求'200字'，回答应控制在180-220字之间；\n"
        "- 如果用户要求'500字'，回答应控制在450-550字之间；\n"
        "- 如果用户要求'简短'或'简要'，回答应控制在100字以内；\n"
        "- 如果用户要求'详细'，回答可以适当展开到300-500字。\n\n"
        "对话上下文（可能为空）：\n{chat_history}\n\n"
        "现在请根据以下背景资料回答用户的问题：\n"
        "{context}"
    )
    # "你是武汉文化助手，根据文档内容回答问题。\n"
    #     "要求：\n"
    #     "- 参考文档内容回答\n"
    #     "- 控制字数：200字=180-220字，500字=450-550字，简短=100字内\n"
    #     "- 直接输出回答，不要思考过程\n\n"
    #     "上下文：\n{chat_history}\n\n"
    #     "文档：\n{context}\n\n"
    #     "问题：{question}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{question}"),
        ]
    )

    # 仅将 question 路由给 retriever，避免把整个 dict 传进去
    chain = (
        {
            "context": itemgetter("question") | retriever | (lambda docs: format_docs_for_prompt(docs)),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def build_streaming_chain():
    """
    构建支持流式输出的RAG链
    """
    load_dotenv()
    setup_logging()

    retriever = load_retriever()
    llm = build_llm()

    system_template = (
        "【核心规则】直接输出答案，禁止任何思考过程、推理步骤、分析说明。\n\n"
        "你是一个熟悉湖北省武汉市和湖北博物馆的历史文化、美食、旅游、风土人情的知识助手，能够结合提供的文档内容和自身知识进行自然、准确的快速地回答。\n\n"
        "你的任务是：\n"
        "- 优先参考我提供的文档内容（即上下文）回答问题；\n"
        "- 如果文档信息不足或缺失，可以适当补充你自身掌握的可靠知识，回答的内容不要过少；\n"
        "- 严格按照用户要求的字数或长度进行回答，如果用户指定了字数（如'200字'、'500字'、'简短回答'等），请严格控制输出长度；\n"
        "- 禁止输出任何思考过程、解释、格式说明或AI语气的语句；\n"
        "- 直接输出自然、口语化的中文回答正文，像一个知识丰富的当地人那样娓娓道来；\n"
        "- 使用纯文本格式输出，禁止使用Markdown格式（如**粗体**、*斜体*、#标题等），不要输出星号、井号、下划线等格式符号；\n"
        "- 请务必保证措辞合理、逻辑通顺、语义信息完整。\n\n"
        "字数控制要求：\n"
        "- 如果没有明确字数要求，默认回答控制在200-300字之间；\n"
        "- 如果用户要求'200字'，回答应控制在180-220字之间；\n"
        "- 如果用户要求'500字'，回答应控制在450-550字之间；\n"
        "- 如果用户要求'简短'或'简要'，回答应控制在100字以内；\n"
        "- 如果用户要求'详细'，回答可以适当展开到300-500字。\n\n"
        "对话上下文（可能为空）：\n{chat_history}\n\n"
        "现在请根据以下背景资料回答用户的问题：\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{question}"),
        ]
    )

    # 构建流式链
    streaming_chain = (
        {
            "context": itemgetter("question") | retriever | (lambda docs: format_docs_for_prompt(docs)),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
    )

    return streaming_chain, retriever

# 在 rag_chain.py 里加一个
def build_generation_chain_only():
    load_dotenv(); setup_logging()
    llm = build_llm()
    system_template = (
        "【核心规则】直接输出答案，禁止任何思考过程、推理步骤、分析说明。\n\n"
        "你是一个熟悉湖北省武汉市和湖北博物馆的历史文化、美食、旅游、风土人情的知识助手，能够结合提供的文档内容和自身知识进行自然、准确的快速地回答。\n\n"
        "你的任务是：\n"
        "- 优先参考我提供的文档内容（即上下文）回答问题；\n"
        "- 如果文档信息不足或缺失，可以适当补充你自身掌握的可靠知识，回答的内容不要过少；\n"
        "- 严格按照用户要求的字数或长度进行回答，如果用户指定了字数（如'200字'、'500字'、'简短回答'等），请严格控制输出长度；\n"
        "- 禁止输出任何思考过程、解释、格式说明或AI语气的语句；\n"
        "- 直接输出自然、口语化的中文回答正文，像一个知识丰富的当地人那样娓娓道来；\n"
        "- 使用纯文本格式输出，禁止使用Markdown格式（如**粗体**、*斜体*、#标题等），不要输出星号、井号、下划线等格式符号；\n"
        "- 请务必保证措辞合理、逻辑通顺、语义信息完整。\n\n"
        "字数控制要求：\n"
        "- 如果没有明确字数要求，默认回答控制在200-300字之间；\n"
        "- 如果用户要求'200字'，回答应控制在200-220字之间；\n"
        "- 如果用户要求'500字'，回答应控制在500-550字之间；\n"
        "- 如果用户要求'简短'或'简要'，回答应控制在100字以内；\n"
        "- 如果用户要求'详细'，回答可以适当展开到300-500字。\n\n"
        "对话上下文（可能为空）：\n{chat_history}\n\n"
        "现在请根据以下背景资料回答用户的问题：\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()


def answer_question(question: str) -> Dict:
    gen_chain = build_generation_chain_only()
    retriever = load_retriever()
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "抱歉，未检索到相关内容。", "sources": []}

    context = format_docs_for_prompt(docs)
    answer = gen_chain.invoke({"question": question, "chat_history": "", "context": context})

    sources = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        chunk_id = d.metadata.get("chunk_id")
        locator = f"page {page}" if page is not None else f"chunk {chunk_id}"
        sources.append({"source": source, "locator": locator})
    return {"answer": answer, "sources": sources}