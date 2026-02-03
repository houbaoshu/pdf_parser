#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Chroma向量数据库的GEA问答Agent
提供高效的向量检索和问答功能
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import chromadb
from chromadb.config import Settings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """文档chunk"""
    id: str
    content: str
    type: str  # text, table, image
    page: int
    source_file: str
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果"""
    chunk: Chunk
    similarity: float
    rank: int


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    query_type: str  # text, image
    search_results: List[SearchResult]
    timestamp: str
    total_chunks_searched: int


class ChromaRetriever:
    """基于Chroma的检索器"""

    def __init__(self, chroma_db_path: str = "chroma_db",
                 collection_name: str = "gea_documents"):
        """
        初始化检索器

        Args:
            chroma_db_path: Chroma数据库路径
            collection_name: 集合名称
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_service = None

    def initialize(self) -> bool:
        """
        初始化Chroma连接和embedding服务

        Returns:
            是否初始化成功
        """
        try:
            # 初始化Chroma客户端
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )

            # 获取集合
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"连接到Chroma集合: {self.collection_name}")
            logger.info(f"集合中共有 {self.collection.count()} 个文档")

        except Exception as e:
            logger.error(f"初始化Chroma失败: {str(e)}")
            return False

        # 初始化embedding服务
        try:
            from embedding_service import create_embedding_service
            self.embedding_service = create_embedding_service()
            logger.info("embedding服务初始化成功")
        except ImportError:
            logger.warning("embedding_service模块未找到，将无法处理文本查询")
            return False

        return True

    def search_by_text(self, query_text: str, top_k: int = 5,
                      chunk_types: Optional[List[str]] = None,
                      source_file: Optional[str] = None,
                      page: Optional[int] = None) -> List[SearchResult]:
        """
        根据文本查询检索相关chunks

        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            chunk_types: 限制的chunk类型列表 (e.g., ["text", "table"])
            source_file: 限制源文件
            page: 限制页码

        Returns:
            搜索结果列表
        """
        if not self.embedding_service:
            logger.error("embedding服务不可用")
            return []

        if not self.collection:
            logger.error("Chroma集合未初始化")
            return []

        # 获取查询文本的embedding
        try:
            embedding_result = self.embedding_service.get_embedding(query_text)
            if embedding_result.error:
                logger.error(f"获取查询embedding失败: {embedding_result.error}")
                return []

            query_embedding = embedding_result.embedding

        except Exception as e:
            logger.error(f"获取查询embedding时发生错误: {str(e)}")
            return []

        # 构建where过滤条件
        where_filter = {}
        if chunk_types:
            if len(chunk_types) == 1:
                where_filter["type"] = chunk_types[0]
            else:
                where_filter["type"] = {"$in": chunk_types}

        if source_file:
            where_filter["source_file"] = source_file

        if page:
            where_filter["page"] = page

        # 执行向量搜索
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            # 解析结果
            search_results = []
            if results and results["ids"]:
                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(results["ids"][0], results["documents"][0],
                        results["metadatas"][0], results["distances"][0])
                ):
                    # 将距离转换为相似度 (Chroma使用L2距离)
                    # L2距离越小越相似，转换为0-1的相似度分数
                    similarity = 1.0 / (1.0 + distance)

                    chunk = Chunk(
                        id=doc_id,
                        content=document,
                        type=metadata.get("type", "text"),
                        page=metadata.get("page", 1),
                        source_file=metadata.get("source_file", ""),
                        char_count=metadata.get("char_count", 0),
                        metadata=metadata
                    )

                    search_results.append(SearchResult(
                        chunk=chunk,
                        similarity=similarity,
                        rank=i + 1
                    ))

            return search_results

        except Exception as e:
            logger.error(f"Chroma查询失败: {str(e)}")
            return []

    def search_by_image_reference(self, image_description: str, top_k: int = 5) -> List[SearchResult]:
        """
        根据图片描述检索相关chunks

        Args:
            image_description: 图片描述
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        # 先搜索图片类型的chunks
        image_results = self.search_by_text(
            image_description,
            top_k=top_k,
            chunk_types=["image"]
        )

        # 如果找到图片，获取图片所在页面的其他chunks
        combined_results = []
        added_ids = set()

        for result in image_results:
            combined_results.append(result)
            added_ids.add(result.chunk.id)

        # 获取图片所在页面的文本chunks
        if image_results:
            for image_result in image_results[:3]:  # 前3个图片结果
                page_results = self.search_by_text(
                    image_description,
                    top_k=5,
                    source_file=image_result.chunk.source_file,
                    page=image_result.chunk.page,
                    chunk_types=["text", "table"]
                )

                for page_result in page_results:
                    if page_result.chunk.id not in added_ids:
                        # 降低相似度
                        page_result.similarity *= 0.8
                        combined_results.append(page_result)
                        added_ids.add(page_result.chunk.id)

        # 重新排序并限制数量
        combined_results.sort(key=lambda x: x.similarity, reverse=True)
        return combined_results[:top_k]

    def search_images_by_keyword(self, keyword: str, top_k: int = 5) -> List[SearchResult]:
        """
        根据关键词搜索图片

        Args:
            keyword: 搜索关键词
            top_k: 返回结果数量

        Returns:
            图片搜索结果列表
        """
        if not self.collection:
            logger.error("Chroma集合未初始化")
            return []

        try:
            # 获取所有图片类型的chunks
            results = self.collection.get(
                where={"type": "image"},
                include=["documents", "metadatas"]
            )

            if not results or not results["ids"]:
                return []

            # 关键词匹配
            matched_results = []
            keyword_lower = keyword.lower()

            for doc_id, document, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                # 检查关键词
                content_lower = document.lower()
                source_file_lower = metadata.get("source_file", "").lower()

                if keyword_lower in content_lower or keyword_lower in source_file_lower:
                    # 计算简单的相似度分数
                    score = 0.0
                    if keyword_lower in content_lower:
                        score += 0.7
                    if keyword_lower in source_file_lower:
                        score += 0.3

                    chunk = Chunk(
                        id=doc_id,
                        content=document,
                        type=metadata.get("type", "image"),
                        page=metadata.get("page", 1),
                        source_file=metadata.get("source_file", ""),
                        char_count=metadata.get("char_count", 0),
                        metadata=metadata
                    )

                    matched_results.append(SearchResult(
                        chunk=chunk,
                        similarity=score,
                        rank=0
                    ))

            # 按相似度排序
            matched_results.sort(key=lambda x: x.similarity, reverse=True)

            # 更新rank
            for i, result in enumerate(matched_results[:top_k]):
                result.rank = i + 1

            return matched_results[:top_k]

        except Exception as e:
            logger.error(f"图片搜索失败: {str(e)}")
            return []

    def get_context_around_chunk(self, chunk: Chunk, context_size: int = 2) -> List[Chunk]:
        """
        获取chunk周围的上下文

        Args:
            chunk: 目标chunk
            context_size: 上下文chunk数量

        Returns:
            上下文chunks列表
        """
        if not self.collection:
            return []

        try:
            # 获取同一页的所有chunks
            results = self.collection.get(
                where={
                    "$and": [
                        {"source_file": chunk.source_file},
                        {"page": chunk.page}
                    ]
                },
                include=["documents", "metadatas"]
            )

            if not results or not results["ids"]:
                return []

            # 转换为Chunk对象
            page_chunks = []
            for doc_id, document, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                page_chunks.append(Chunk(
                    id=doc_id,
                    content=document,
                    type=metadata.get("type", "text"),
                    page=metadata.get("page", 1),
                    source_file=metadata.get("source_file", ""),
                    char_count=metadata.get("char_count", 0),
                    metadata=metadata
                ))

            # 找到目标chunk的位置
            target_index = -1
            for i, c in enumerate(page_chunks):
                if c.id == chunk.id:
                    target_index = i
                    break

            if target_index == -1:
                return []

            # 获取上下文范围
            start_idx = max(0, target_index - context_size)
            end_idx = min(len(page_chunks), target_index + context_size + 1)

            return page_chunks[start_idx:end_idx]

        except Exception as e:
            logger.error(f"获取上下文失败: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息

        Returns:
            统计信息字典
        """
        if not self.collection:
            return {}

        try:
            total_count = self.collection.count()

            # 获取样本来统计类型
            sample = self.collection.get(limit=min(total_count, 1000))
            type_counts = {}

            if sample and "metadatas" in sample:
                for meta in sample["metadatas"]:
                    chunk_type = meta.get("type", "unknown")
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

            return {
                "total_chunks": total_count,
                "type_distribution": type_counts,
                "sample_size": len(sample["ids"]) if sample else 0
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}


class GEAQAAgentChroma:
    """基于Chroma的GEA问答Agent"""

    def __init__(self, chroma_db_path: str = "chroma_db",
                 collection_name: str = "gea_documents"):
        """
        初始化问答Agent

        Args:
            chroma_db_path: Chroma数据库路径
            collection_name: 集合名称
        """
        self.retriever = ChromaRetriever(chroma_db_path, collection_name)
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        初始化Agent

        Returns:
            是否初始化成功
        """
        logger.info("正在初始化基于Chroma的GEA问答Agent...")

        if not self.retriever.initialize():
            logger.error("初始化检索器失败")
            return False

        self.is_initialized = True

        # 显示统计信息
        stats = self.retriever.get_statistics()
        logger.info("GEA问答Agent初始化成功")
        logger.info(f"可用chunks: {stats.get('total_chunks', 0)}")
        if "type_distribution" in stats:
            for chunk_type, count in stats["type_distribution"].items():
                logger.info(f"{chunk_type} chunks: {count}")

        return True

    def query(self, question: str, query_type: str = "text", top_k: int = 5,
             chunk_types: Optional[List[str]] = None) -> QueryResult:
        """
        执行查询

        Args:
            question: 问题或查询内容
            query_type: 查询类型 (text, image)
            top_k: 返回结果数量
            chunk_types: 限制的chunk类型列表

        Returns:
            查询结果
        """
        if not self.is_initialized:
            logger.error("Agent未初始化，请先调用initialize()")
            return QueryResult(
                query=question,
                query_type=query_type,
                search_results=[],
                timestamp=datetime.now().isoformat(),
                total_chunks_searched=0
            )

        logger.info(f"执行{query_type}查询: {question}")

        # 根据查询类型执行检索
        if query_type == "text":
            search_results = self.retriever.search_by_text(
                question, top_k=top_k, chunk_types=chunk_types
            )
        elif query_type == "image":
            search_results = self.retriever.search_by_image_reference(question, top_k=top_k)
        else:
            logger.error(f"不支持的查询类型: {query_type}")
            search_results = []

        # 获取总文档数
        stats = self.retriever.get_statistics()
        total_chunks = stats.get("total_chunks", 0)

        # 创建查询结果
        result = QueryResult(
            query=question,
            query_type=query_type,
            search_results=search_results,
            timestamp=datetime.now().isoformat(),
            total_chunks_searched=total_chunks
        )

        logger.info(f"查询完成，找到 {len(search_results)} 个相关结果")

        return result

    def search_images(self, keyword: str, top_k: int = 5) -> List[SearchResult]:
        """
        搜索图片

        Args:
            keyword: 搜索关键词
            top_k: 返回结果数量

        Returns:
            图片搜索结果
        """
        if not self.is_initialized:
            logger.error("Agent未初始化")
            return []

        return self.retriever.search_images_by_keyword(keyword, top_k=top_k)

    def get_answer_with_context(self, question: str, include_context: bool = True) -> Dict[str, Any]:
        """
        获取带上下文的答案

        Args:
            question: 问题
            include_context: 是否包含上下文

        Returns:
            答案字典
        """
        # 执行查询
        query_result = self.query(question, query_type="text", top_k=3)

        if not query_result.search_results:
            return {
                "question": question,
                "answer": "抱歉，没有找到相关信息。",
                "sources": [],
                "has_answer": False
            }

        # 构建答案
        top_result = query_result.search_results[0]
        answer_parts = []
        sources = []

        # 使用最相关的结果作为主要答案
        main_chunk = top_result.chunk
        answer_parts.append(main_chunk.content)
        sources.append({
            "source_file": main_chunk.source_file,
            "page": main_chunk.page,
            "type": main_chunk.type,
            "similarity": top_result.similarity
        })

        # 添加上下文
        if include_context:
            context_chunks = self.retriever.get_context_around_chunk(main_chunk, context_size=1)
            for context_chunk in context_chunks:
                if context_chunk.id != main_chunk.id:
                    answer_parts.append(context_chunk.content)

        # 组合答案
        answer = " ".join(answer_parts)

        # 如果答案太长，截断
        if len(answer) > 1000:
            answer = answer[:1000] + "..."

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "has_answer": True,
            "confidence": top_result.similarity
        }

    def get_image_info(self, image_keyword: str) -> Dict[str, Any]:
        """
        获取图片信息

        Args:
            image_keyword: 图片关键词

        Returns:
            图片信息字典
        """
        # 搜索图片
        image_results = self.search_images(image_keyword, top_k=3)

        if not image_results:
            return {
                "keyword": image_keyword,
                "found": False,
                "message": "未找到相关图片"
            }

        # 获取图片信息
        images_info = []
        for result in image_results:
            chunk = result.chunk
            images_info.append({
                "source_file": chunk.source_file,
                "page": chunk.page,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "similarity": result.similarity
            })

        return {
            "keyword": image_keyword,
            "found": True,
            "images": images_info,
            "total_found": len(image_results)
        }


def print_query_result(result: QueryResult):
    """打印查询结果"""
    print(f"\n查询: {result.query}")
    print(f"类型: {result.query_type}")
    print(f"时间: {result.timestamp}")
    print(f"搜索chunks数: {result.total_chunks_searched}")
    print(f"相关结果数: {len(result.search_results)}")
    print("-" * 80)

    for i, search_result in enumerate(result.search_results):
        chunk = search_result.chunk
        print(f"\n{i+1}. [相似度: {search_result.similarity:.4f}]")
        print(f"   类型: {chunk.type}")
        print(f"   来源: {chunk.source_file}")
        print(f"   页码: {chunk.page}")
        content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        print(f"   内容预览: {content_preview}")


def main():
    """主函数"""
    print("=" * 80)
    print("基于Chroma的GEA多模态文档问答Agent")
    print("=" * 80)

    # 创建Agent
    agent = GEAQAAgentChroma()

    # 初始化
    print("\n正在初始化Agent...")
    if not agent.initialize():
        print("初始化失败，请检查Chroma数据库")
        return

    print("\nAgent初始化成功!")

    # 示例查询
    print("\n" + "=" * 80)
    print("示例查询")
    print("=" * 80)

    # 示例1: 文本查询
    print("\n1. 文本查询示例:")
    text_query = "GEA设备技术参数"
    result1 = agent.query(text_query, query_type="text", top_k=3)
    print_query_result(result1)

    # 示例2: 获取带上下文的答案
    print("\n\n2. 带上下文的答案示例:")
    question = "GEA设备的主要特点是什么？"
    answer = agent.get_answer_with_context(question)
    print(f"问题: {answer['question']}")
    print(f"答案: {answer['answer']}")
    print(f"置信度: {answer.get('confidence', 0):.4f}")
    if answer['sources']:
        print("来源:")
        for source in answer['sources']:
            print(f"  - {source['source_file']} (页{source['page']})")

    # 示例3: 图片查询
    print("\n\n3. 图片查询示例:")
    image_query = "设备结构图"
    result3 = agent.query(image_query, query_type="image", top_k=2)
    print_query_result(result3)

    # 示例4: 图片搜索
    print("\n\n4. 图片搜索示例:")
    image_keyword = "图表"
    image_info = agent.get_image_info(image_keyword)
    print(f"搜索关键词: {image_info['keyword']}")
    print(f"找到图片: {image_info['found']}")
    if image_info['found']:
        print(f"找到 {image_info['total_found']} 张图片:")
        for i, img in enumerate(image_info['images']):
            print(f"\n{i+1}. {img['source_file']} (页{img['page']})")
            print(f"   描述: {img['content']}")
            print(f"   相似度: {img['similarity']:.4f}")

    print("\n" + "=" * 80)
    print("Agent测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
