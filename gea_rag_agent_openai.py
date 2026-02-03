#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的RAG问答Agent - 集成Chroma检索 + OpenAI答案生成
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI

# 导入Chroma检索器
from gea_qa_agent_chroma import GEAQAAgentChroma, SearchResult, Chunk

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """RAG答案"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    search_results: List[SearchResult]
    timestamp: str
    model: str
    tokens_used: Optional[int] = None
    has_answer: bool = True


class GEARAGAgent:
    """基于Chroma检索和OpenAI生成的RAG Agent"""

    def __init__(self,
                 chroma_db_path: str = "chroma_db",
                 collection_name: str = "gea_documents",
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None):
        """
        初始化RAG Agent

        Args:
            chroma_db_path: Chroma数据库路径
            collection_name: 集合名称
            model: OpenAI模型名称
            api_key: OpenAI API密钥（如果不提供则从环境变量读取）
        """
        self.model = model
        self.qa_agent = GEAQAAgentChroma(chroma_db_path, collection_name)

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self.is_initialized = False

    def initialize(self) -> bool:
        """
        初始化Agent

        Returns:
            是否初始化成功
        """
        logger.info("正在初始化RAG Agent...")

        # 初始化检索Agent
        if not self.qa_agent.initialize():
            logger.error("初始化检索Agent失败")
            return False

        # 测试OpenAI连接
        try:
            # 简单测试
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            logger.info(f"OpenAI API连接成功，使用模型: {self.model}")
        except Exception as e:
            logger.error(f"OpenAI API连接失败: {str(e)}")
            return False

        self.is_initialized = True
        logger.info("RAG Agent初始化成功")
        return True

    def query(self,
             question: str,
             top_k: int = 5,
             chunk_types: Optional[List[str]] = None,
             include_context: bool = True,
             temperature: float = 0.7,
             max_tokens: int = 1000) -> RAGAnswer:
        """
        执行RAG查询 - 检索相关文档并生成答案

        Args:
            question: 用户问题
            top_k: 检索的文档数量
            chunk_types: 限制的chunk类型
            include_context: 是否包含上下文chunks
            temperature: 生成温度
            max_tokens: 最大生成token数

        Returns:
            RAG答案
        """
        if not self.is_initialized:
            logger.error("Agent未初始化")
            return RAGAnswer(
                question=question,
                answer="系统未初始化，请先调用initialize()",
                sources=[],
                search_results=[],
                timestamp=datetime.now().isoformat(),
                model=self.model,
                has_answer=False
            )

        logger.info(f"处理问题: {question}")

        # 1. 检索相关文档
        logger.info(f"检索top {top_k}相关文档...")
        query_result = self.qa_agent.query(
            question,
            query_type="text",
            top_k=top_k,
            chunk_types=chunk_types
        )

        if not query_result.search_results:
            logger.warning("未找到相关文档")
            return RAGAnswer(
                question=question,
                answer="抱歉，我没有找到与您问题相关的信息。",
                sources=[],
                search_results=[],
                timestamp=datetime.now().isoformat(),
                model=self.model,
                has_answer=False
            )

        # 2. 构建context
        logger.info("构建上下文...")
        context = self._build_context(
            query_result.search_results,
            include_context=include_context
        )

        # 3. 生成答案
        logger.info("使用OpenAI生成答案...")
        answer, tokens_used = self._generate_answer(
            question=question,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 4. 提取来源信息
        sources = []
        for result in query_result.search_results:
            sources.append({
                "source_file": result.chunk.source_file,
                "page": result.chunk.page,
                "type": result.chunk.type,
                "similarity": result.similarity,
                "content_preview": result.chunk.content[:100] + "..." if len(result.chunk.content) > 100 else result.chunk.content
            })

        logger.info(f"答案生成完成，使用了 {tokens_used} tokens")

        return RAGAnswer(
            question=question,
            answer=answer,
            sources=sources,
            search_results=query_result.search_results,
            timestamp=datetime.now().isoformat(),
            model=self.model,
            tokens_used=tokens_used,
            has_answer=True
        )

    def _build_context(self,
                      search_results: List[SearchResult],
                      include_context: bool = True,
                      max_chunks: int = 10) -> str:
        """
        构建上下文字符串

        Args:
            search_results: 搜索结果列表
            include_context: 是否包含周边chunks
            max_chunks: 最大chunk数量

        Returns:
            上下文字符串
        """
        context_parts = []
        added_chunks = set()

        for i, result in enumerate(search_results[:max_chunks]):
            chunk = result.chunk

            # 添加主chunk
            if chunk.id not in added_chunks:
                context_parts.append(self._format_chunk(chunk, i + 1, result.similarity))
                added_chunks.add(chunk.id)

            # 添加上下文chunks
            if include_context and i < 3:  # 只为前3个结果添加上下文
                context_chunks = self.qa_agent.retriever.get_context_around_chunk(
                    chunk, context_size=1
                )
                for ctx_chunk in context_chunks:
                    if ctx_chunk.id not in added_chunks and len(added_chunks) < max_chunks:
                        context_parts.append(self._format_chunk(ctx_chunk, None, None))
                        added_chunks.add(ctx_chunk.id)

        return "\n\n".join(context_parts)

    def _format_chunk(self,
                     chunk: Chunk,
                     rank: Optional[int] = None,
                     similarity: Optional[float] = None) -> str:
        """
        格式化chunk为上下文字符串

        Args:
            chunk: Chunk对象
            rank: 排名
            similarity: 相似度

        Returns:
            格式化的字符串
        """
        header_parts = []

        if rank:
            header_parts.append(f"[文档{rank}]")

        header_parts.append(f"来源: {os.path.basename(chunk.source_file)}")
        header_parts.append(f"页码: {chunk.page}")
        header_parts.append(f"类型: {chunk.type}")

        if similarity:
            header_parts.append(f"相似度: {similarity:.3f}")

        header = " | ".join(header_parts)

        # 处理content
        content = chunk.content
        if chunk.type == "table" and content.startswith("["):
            # 表格内容，尝试格式化
            try:
                import json
                table_data = json.loads(content)
                content = "表格内容：\n" + self._format_table(table_data)
            except:
                pass

        return f"{header}\n{content}"

    def _format_table(self, table_data: List[List[Any]], max_rows: int = 10) -> str:
        """
        格式化表格数据

        Args:
            table_data: 表格数据
            max_rows: 最大行数

        Returns:
            格式化的表格字符串
        """
        if not table_data:
            return ""

        lines = []
        for i, row in enumerate(table_data[:max_rows]):
            row_str = " | ".join([str(cell) if cell else "" for cell in row])
            lines.append(row_str)

        if len(table_data) > max_rows:
            lines.append(f"... (还有 {len(table_data) - max_rows} 行)")

        return "\n".join(lines)

    def _generate_answer(self,
                        question: str,
                        context: str,
                        temperature: float = 0.7,
                        max_tokens: int = 1000) -> tuple[str, int]:
        """
        使用OpenAI生成答案

        Args:
            question: 用户问题
            context: 上下文
            temperature: 生成温度
            max_tokens: 最大token数

        Returns:
            (答案, 使用的tokens数)
        """
        # 构建系统提示
        system_prompt = """你是一个专业的GEA设备技术文档助手。你的任务是根据提供的文档内容回答用户关于GEA设备的问题。

回答要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户
3. 回答要准确、简洁、专业
4. 对于技术参数，要准确引用文档中的数据
5. 如果涉及多个文档片段，要综合归纳
6. 使用中文回答

如果文档中包含表格数据，请准确提取和总结相关信息。"""

        # 构建用户消息
        user_message = f"""参考文档：
{context}

---

用户问题：{question}

请根据上述文档内容回答问题。如果文档中没有相关信息，请明确说明。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            return answer, tokens_used

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            return f"抱歉，生成答案时发生错误: {str(e)}", 0

    def chat(self,
            question: str,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            top_k: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 1000) -> tuple[str, List[Dict[str, str]]]:
        """
        对话模式 - 支持多轮对话

        Args:
            question: 用户问题
            conversation_history: 对话历史 [{"role": "user", "content": "..."}, ...]
            top_k: 检索文档数量
            temperature: 生成温度
            max_tokens: 最大token数

        Returns:
            (答案, 更新后的对话历史)
        """
        # 执行RAG查询
        rag_answer = self.query(
            question=question,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 更新对话历史
        if conversation_history is None:
            conversation_history = []

        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": rag_answer.answer})

        return rag_answer.answer, conversation_history


def print_rag_answer(answer: RAGAnswer):
    """打印RAG答案"""
    print("\n" + "=" * 80)
    print(f"问题: {answer.question}")
    print("=" * 80)
    print(f"\n答案:\n{answer.answer}")
    print("\n" + "-" * 80)
    print(f"来源文档 ({len(answer.sources)}个):")
    for i, source in enumerate(answer.sources, 1):
        print(f"\n{i}. {source['source_file']} - 第{source['page']}页")
        print(f"   类型: {source['type']} | 相似度: {source['similarity']:.3f}")
        print(f"   内容: {source['content_preview']}")

    print("\n" + "-" * 80)
    print(f"模型: {answer.model}")
    if answer.tokens_used:
        print(f"Tokens使用: {answer.tokens_used}")
    print(f"时间: {answer.timestamp}")
    print("=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("GEA RAG问答系统 - Chroma + OpenAI")
    print("=" * 80)

    # 创建Agent
    agent = GEARAGAgent(model="gpt-4o-mini")

    # 初始化
    print("\n正在初始化RAG Agent...")
    if not agent.initialize():
        print("初始化失败")
        return

    print("\n✅ RAG Agent初始化成功！")

    # 示例问题
    test_questions = [
        "GEA设备的主要技术参数有哪些？",
        "TPS系列泵的结构尺寸和转速是多少？",
        "如何维护和保养GEA设备？",
    ]

    print("\n" + "=" * 80)
    print("示例问答")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'='*80}")
        print(f"示例 {i}/{len(test_questions)}")
        print(f"{'='*80}")

        answer = agent.query(
            question=question,
            top_k=3,
            temperature=0.7
        )

        print_rag_answer(answer)

    print("\n\n" + "=" * 80)
    print("✅ RAG系统测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
