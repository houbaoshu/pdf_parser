#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEA多模态文档问答Agent
支持文本和图片查询，检索embeddings/GEA下的embedding数据
"""

import os
import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

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
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    char_count: int = 0
    block_indices: List[int] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)


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


class GEAEmbeddingLoader:
    """GEA embedding数据加载器"""

    def __init__(self, embeddings_dir: str = "embeddings/GEA"):
        """
        初始化加载器

        Args:
            embeddings_dir: embedding数据目录
        """
        self.embeddings_dir = embeddings_dir
        self.chunks: List[Chunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.chunk_id_to_index: Dict[str, int] = {}

    def load_all_embeddings(self) -> bool:
        """
        加载所有embedding数据

        Returns:
            是否加载成功
        """
        if not os.path.exists(self.embeddings_dir):
            logger.error(f"embedding目录不存在: {self.embeddings_dir}")
            return False

        # 查找所有embedding文件
        embedding_files = []
        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_embeddings.json") and file != "processing_summary.json":
                embedding_files.append(os.path.join(self.embeddings_dir, file))

        if not embedding_files:
            logger.error(f"在 {self.embeddings_dir} 中未找到embedding文件")
            return False

        logger.info(f"找到 {len(embedding_files)} 个embedding文件")

        # 加载所有chunks
        all_chunks = []
        for file_path in embedding_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取chunks
                chunks_data = data.get("chunks", [])
                source_file = data.get("pdf_path", "")

                for i, chunk_data in enumerate(chunks_data):
                    # 生成唯一ID
                    chunk_id = hashlib.md5(f"{source_file}_{i}".encode()).hexdigest()[:16]

                    # 创建Chunk对象
                    chunk = Chunk(
                        id=chunk_id,
                        content=chunk_data.get("content", ""),
                        type=chunk_data.get("type", "text"),
                        page=chunk_data.get("page", 1),
                        embedding=chunk_data.get("embedding", {}).get("vector", []),
                        metadata=chunk_data.get("metadata", {}),
                        source_file=source_file,
                        char_count=chunk_data.get("char_count", 0),
                        block_indices=chunk_data.get("block_indices", []),
                        blocks=chunk_data.get("blocks", [])
                    )

                    all_chunks.append(chunk)

            except Exception as e:
                logger.error(f"加载embedding文件失败 {file_path}: {str(e)}")
                continue

        if not all_chunks:
            logger.error("未加载到任何chunks")
            return False

        self.chunks = all_chunks
        logger.info(f"成功加载 {len(self.chunks)} 个chunks")

        # 构建embedding矩阵
        self._build_embeddings_matrix()

        return True

    def _build_embeddings_matrix(self):
        """构建embedding矩阵"""
        if not self.chunks:
            return

        # 提取所有embedding向量
        embeddings = []
        valid_chunks = []

        for i, chunk in enumerate(self.chunks):
            if chunk.embedding and len(chunk.embedding) > 0:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
                self.chunk_id_to_index[chunk.id] = len(valid_chunks) - 1

        if embeddings:
            self.embeddings_matrix = np.array(embeddings)
            self.chunks = valid_chunks  # 只保留有embedding的chunks
            logger.info(f"构建embedding矩阵: {self.embeddings_matrix.shape}")
        else:
            logger.warning("没有有效的embedding向量")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """根据ID获取chunk"""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_chunks_by_type(self, chunk_type: str) -> List[Chunk]:
        """根据类型获取chunks"""
        return [chunk for chunk in self.chunks if chunk.type == chunk_type]

    def get_chunks_by_page(self, page: int, source_file: Optional[str] = None) -> List[Chunk]:
        """根据页码获取chunks"""
        if source_file:
            return [chunk for chunk in self.chunks if chunk.page == page and chunk.source_file == source_file]
        else:
            return [chunk for chunk in self.chunks if chunk.page == page]


class GEARetriever:
    """GEA检索器"""

    def __init__(self, embedding_loader: GEAEmbeddingLoader):
        """
        初始化检索器

        Args:
            embedding_loader: embedding数据加载器
        """
        self.loader = embedding_loader
        self.embedding_service = None

        # 尝试导入embedding服务
        try:
            from embedding_service import create_embedding_service, EmbeddingConfig
            self.embedding_service = create_embedding_service()
            logger.info("embedding服务初始化成功")
        except ImportError:
            logger.warning("embedding_service模块未找到，将无法处理文本查询")

    def search_by_text(self, query_text: str, top_k: int = 5,
                      chunk_types: Optional[List[str]] = None) -> List[SearchResult]:
        """
        根据文本查询检索相关chunks

        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            chunk_types: 限制的chunk类型列表

        Returns:
            搜索结果列表
        """
        if not self.embedding_service:
            logger.error("embedding服务不可用，无法处理文本查询")
            return []

        if self.loader.embeddings_matrix is None or self.loader.embeddings_matrix.size == 0:
            logger.error("embedding矩阵未初始化")
            return []

        # 获取查询文本的embedding
        try:
            embedding_result = self.embedding_service.get_embedding(query_text)
            if embedding_result.error:
                logger.error(f"获取查询embedding失败: {embedding_result.error}")
                return []

            query_embedding = np.array(embedding_result.embedding)

        except Exception as e:
            logger.error(f"获取查询embedding时发生错误: {str(e)}")
            return []

        # 计算相似度
        similarities = self._calculate_similarities(query_embedding)

        # 过滤和排序结果
        results = []
        for i, similarity in enumerate(similarities):
            chunk = self.loader.chunks[i]

            # 类型过滤
            if chunk_types and chunk.type not in chunk_types:
                continue

            results.append(SearchResult(
                chunk=chunk,
                similarity=similarity,
                rank=len(results) + 1
            ))

        # 按相似度排序
        results.sort(key=lambda x: x.similarity, reverse=True)

        # 返回top_k结果
        return results[:top_k]

    def search_by_image_reference(self, image_description: str, top_k: int = 5) -> List[SearchResult]:
        """
        根据图片描述检索相关chunks

        Args:
            image_description: 图片描述
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        # 首先搜索图片相关的chunks
        image_chunks = self.loader.get_chunks_by_type("image")

        if not image_chunks:
            logger.warning("未找到图片chunks")
            return []

        # 使用图片描述进行文本搜索
        search_results = self.search_by_text(image_description, top_k=top_k * 2)

        # 优先返回包含图片的chunks
        image_results = []
        other_results = []

        for result in search_results:
            if result.chunk.type == "image":
                image_results.append(result)
            else:
                other_results.append(result)

        # 组合结果：先图片chunks，再其他chunks
        combined_results = image_results + other_results

        # 如果找到图片chunks，也返回图片所在页面的其他chunks
        if image_results:
            for image_result in image_results[:3]:  # 前3个图片结果
                page = image_result.chunk.page
                source_file = image_result.chunk.source_file

                # 获取同一页的其他chunks
                page_chunks = self.loader.get_chunks_by_page(page, source_file)
                for chunk in page_chunks:
                    if chunk.type != "image" and chunk.id not in [r.chunk.id for r in combined_results]:
                        combined_results.append(SearchResult(
                            chunk=chunk,
                            similarity=image_result.similarity * 0.8,  # 降低相似度
                            rank=len(combined_results) + 1
                        ))

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
        # 获取所有图片chunks
        image_chunks = self.loader.get_chunks_by_type("image")

        if not image_chunks:
            return []

        # 简单的关键词匹配
        results = []
        keyword_lower = keyword.lower()

        for chunk in image_chunks:
            # 检查chunk内容是否包含关键词
            content_lower = chunk.content.lower()
            metadata_str = str(chunk.metadata).lower()

            if (keyword_lower in content_lower or
                keyword_lower in metadata_str or
                keyword_lower in chunk.source_file.lower()):

                # 计算简单的相似度分数
                score = 0.0
                if keyword_lower in content_lower:
                    score += 0.5
                if keyword_lower in metadata_str:
                    score += 0.3
                if keyword_lower in chunk.source_file.lower():
                    score += 0.2

                results.append(SearchResult(
                    chunk=chunk,
                    similarity=score,
                    rank=len(results) + 1
                ))

        # 按相似度排序
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        计算查询embedding与所有chunks的相似度

        Args:
            query_embedding: 查询embedding向量

        Returns:
            相似度数组
        """
        if self.loader.embeddings_matrix is None:
            return np.array([])

        # 余弦相似度计算
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(self.loader.chunks))

        # 归一化embedding矩阵
        embeddings_norm = np.linalg.norm(self.loader.embeddings_matrix, axis=1)
        valid_mask = embeddings_norm > 0

        if not np.any(valid_mask):
            return np.zeros(len(self.loader.chunks))

        # 计算余弦相似度
        similarities = np.zeros(len(self.loader.chunks))
        similarities[valid_mask] = np.dot(
            self.loader.embeddings_matrix[valid_mask],
            query_embedding
        ) / (embeddings_norm[valid_mask] * query_norm)

        return similarities

    def get_context_around_chunk(self, chunk: Chunk, context_size: int = 2) -> List[Chunk]:
        """
        获取chunk周围的上下文

        Args:
            chunk: 目标chunk
            context_size: 上下文chunk数量

        Returns:
            上下文chunks列表
        """
        # 获取同一文件同一页的所有chunks
        page_chunks = self.loader.get_chunks_by_page(chunk.page, chunk.source_file)

        if not page_chunks:
            return []

        # 找到目标chunk在页面中的位置
        try:
            chunk_index = page_chunks.index(chunk)
        except ValueError:
            return []

        # 获取上下文范围
        start_idx = max(0, chunk_index - context_size)
        end_idx = min(len(page_chunks), chunk_index + context_size + 1)

        return page_chunks[start_idx:end_idx]


class GEAQAAgent:
    """GEA问答Agent"""

    def __init__(self, embeddings_dir: str = "embeddings/GEA"):
        """
        初始化问答Agent

        Args:
            embeddings_dir: embedding数据目录
        """
        self.embeddings_dir = embeddings_dir
        self.loader = GEAEmbeddingLoader(embeddings_dir)
        self.retriever = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        初始化Agent

        Returns:
            是否初始化成功
        """
        logger.info("正在初始化GEA问答Agent...")

        # 加载embedding数据
        if not self.loader.load_all_embeddings():
            logger.error("加载embedding数据失败")
            return False

        # 初始化检索器
        self.retriever = GEARetriever(self.loader)

        self.is_initialized = True
        logger.info("GEA问答Agent初始化成功")
        logger.info(f"可用chunks: {len(self.loader.chunks)}")
        logger.info(f"文本chunks: {len(self.loader.get_chunks_by_type('text'))}")
        logger.info(f"表格chunks: {len(self.loader.get_chunks_by_type('table'))}")
        logger.info(f"图片chunks: {len(self.loader.get_chunks_by_type('image'))}")

        return True

    def query(self, question: str, query_type: str = "text", top_k: int = 5) -> QueryResult:
        """
        执行查询

        Args:
            question: 问题或查询内容
            query_type: 查询类型 (text, image)
            top_k: 返回结果数量

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
            search_results = self.retriever.search_by_text(question, top_k=top_k)
        elif query_type == "image":
            search_results = self.retriever.search_by_image_reference(question, top_k=top_k)
        else:
            logger.error(f"不支持的查询类型: {query_type}")
            search_results = []

        # 创建查询结果
        result = QueryResult(
            query=question,
            query_type=query_type,
            search_results=search_results,
            timestamp=datetime.now().isoformat(),
            total_chunks_searched=len(self.loader.chunks)
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
        print(f"   内容预览: {chunk.content[:100]}..." if len(chunk.content) > 100 else f"   内容: {chunk.content}")
        print(f"   字符数: {chunk.char_count}")


def main():
    """主函数"""
    print("=" * 80)
    print("GEA多模态文档问答Agent")
    print("=" * 80)

    # 创建Agent
    agent = GEAQAAgent()

    # 初始化
    print("\n正在初始化Agent...")
    if not agent.initialize():
        print("初始化失败，请检查embedding数据")
        return

    print(f"\nAgent初始化成功!")
    print(f"加载chunks: {len(agent.loader.chunks)}")

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

