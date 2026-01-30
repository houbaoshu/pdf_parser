#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地embedding服务模块
使用bge-base-zh-v1.5等本地模型生成embedding
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """支持的本地embedding模型"""
    BGE_BASE_ZH_V1_5 = "BAAI/bge-base-zh-v1.5"
    BGE_LARGE_ZH_V1_5 = "BAAI/bge-large-zh-v1.5"


@dataclass
class EmbeddingConfig:
    """本地Embedding配置"""
    # 模型配置
    model: EmbeddingModel = EmbeddingModel.BGE_BASE_ZH_V1_5
    local_model_name: str = "BAAI/bge-base-zh-v1.5"  # 本地模型名称
    device: str = "cpu"  # 设备：cpu或cuda
    normalize_embeddings: bool = True  # 是否归一化embedding

    # 请求配置
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100  # 批量处理的大小


@dataclass
class EmbeddingResult:
    """Embedding结果"""
    embedding: List[float]
    model: str
    dimensions: int
    token_count: Optional[int] = None
    error: Optional[str] = None


class EmbeddingService:
    """本地Embedding服务"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化embedding服务

        Args:
            config: Embedding配置，如果为None则使用默认配置
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._initialize_model()

        logger.info(f"初始化本地embedding服务，模型: {self.config.local_model_name}")

    def _initialize_model(self):
        """初始化本地模型"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"正在加载本地模型: {self.config.local_model_name}")
            logger.info(f"使用设备: {self.config.device}")

            self._model = SentenceTransformer(
                self.config.local_model_name,
                device=self.config.device
            )

            logger.info(f"本地模型加载成功: {self.config.local_model_name}")
            logger.info(f"模型维度: {self._model.get_sentence_embedding_dimension()}")

        except ImportError as e:
            logger.error("需要安装sentence-transformers库: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载本地模型失败: {str(e)}")
            raise

    def get_embedding(self, text: str) -> EmbeddingResult:
        """
        获取单个文本的embedding

        Args:
            text: 输入文本

        Returns:
            EmbeddingResult对象
        """
        if not text or not text.strip():
            return EmbeddingResult(
                embedding=[],
                model=self.config.local_model_name,
                dimensions=0,
                error="输入文本为空"
            )

        try:
            # 使用本地模型生成embedding
            embedding = self._model.encode(
                [text],
                normalize_embeddings=self.config.normalize_embeddings
            )[0].tolist()

            return EmbeddingResult(
                embedding=embedding,
                model=self.config.local_model_name,
                dimensions=len(embedding),
                token_count=len(text)
            )

        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            return EmbeddingResult(
                embedding=[],
                model=self.config.local_model_name,
                dimensions=0,
                error=str(e)
            )

    def get_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        批量获取文本的embedding

        Args:
            texts: 文本列表

        Returns:
            EmbeddingResult列表
        """
        if not texts:
            return []

        if not self._model:
            return [EmbeddingResult([], "", 0, error="模型未初始化") for _ in texts]

        try:
            # 分批处理
            results = []
            batch_size = self.config.batch_size

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"处理embedding批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                try:
                    # 使用本地模型批量生成embedding
                    embeddings = self._model.encode(
                        batch_texts,
                        normalize_embeddings=self.config.normalize_embeddings
                    )

                    # 创建结果
                    for j, embedding in enumerate(embeddings):
                        text_idx = i + j
                        if text_idx < len(texts):
                            results.append(EmbeddingResult(
                                embedding=embedding.tolist(),
                                model=self.config.local_model_name,
                                dimensions=len(embedding),
                                token_count=len(texts[text_idx]) if texts[text_idx] else 0
                            ))

                except Exception as e:
                    logger.error(f"批量获取embedding失败: {str(e)}")
                    # 为失败的批次创建错误结果
                    for text in batch_texts:
                        results.append(EmbeddingResult(
                            embedding=[],
                            model=self.config.local_model_name,
                            dimensions=0,
                            error=str(e)
                        ))

            return results

        except Exception as e:
            logger.error(f"批量获取embedding失败: {str(e)}")
            return [EmbeddingResult([], "", 0, error=str(e)) for _ in texts]

    def get_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为chunks获取embedding

        Args:
            chunks: chunk列表，每个chunk是包含content字段的字典

        Returns:
            添加了embedding信息的chunk列表
        """
        if not chunks:
            return []

        # 提取文本内容
        texts = []
        for chunk in chunks:
            if chunk.get("type") == "text":
                content = chunk.get("content", "")
            elif chunk.get("type") == "table":
                content = chunk.get("text_content", "")
            elif chunk.get("type") == "image":
                content = chunk.get("content", "")
            else:
                content = ""

            texts.append(content)

        # 获取embedding
        embedding_results = self.get_embeddings_batch(texts)

        # 将embedding结果添加到chunks中
        for i, (chunk, embedding_result) in enumerate(zip(chunks, embedding_results)):
            chunk["embedding"] = {
                "vector": embedding_result.embedding,
                "model": embedding_result.model,
                "dimensions": embedding_result.dimensions,
                "token_count": embedding_result.token_count,
                "has_error": embedding_result.error is not None,
                "error": embedding_result.error
            }

            # 记录成功/失败
            if embedding_result.error:
                logger.warning(f"Chunk {i+1} embedding失败: {embedding_result.error}")
            else:
                logger.debug(f"Chunk {i+1} embedding成功，维度: {embedding_result.dimensions}")

        return chunks

    def validate_config(self) -> bool:
        """
        验证配置是否有效

        Returns:
            配置是否有效
        """
        try:
            test_result = self.get_embedding("test")
            if test_result.error:
                logger.error(f"配置验证失败: {test_result.error}")
                return False

            logger.info(f"配置验证成功，模型: {test_result.model}，维度: {test_result.dimensions}")
            return True

        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False

    def close(self):
        """关闭服务"""
        if self._model:
            # 清理模型资源
            del self._model
            self._model = None
            logger.info("本地模型资源已释放")


def load_config_from_env() -> EmbeddingConfig:
    """
    从环境变量加载配置

    Returns:
        EmbeddingConfig对象
    """
    config = EmbeddingConfig()

    # 从环境变量获取本地模型名称
    local_model_name = os.getenv("EMBEDDING_LOCAL_MODEL", "")
    if local_model_name:
        config.local_model_name = local_model_name

    # 从环境变量获取设备
    device = os.getenv("EMBEDDING_DEVICE", "")
    if device:
        config.device = device

    # 从环境变量获取其他配置
    timeout = os.getenv("EMBEDDING_TIMEOUT", "")
    if timeout:
        try:
            config.timeout = int(timeout)
        except ValueError:
            logger.warning(f"无效的timeout: {timeout}")

    batch_size = os.getenv("EMBEDDING_BATCH_SIZE", "")
    if batch_size:
        try:
            config.batch_size = int(batch_size)
        except ValueError:
            logger.warning(f"无效的batch_size: {batch_size}")

    return config


def create_embedding_service(config: Optional[EmbeddingConfig] = None) -> EmbeddingService:
    """
    创建embedding服务

    Args:
        config: 配置对象，如果为None则从环境变量加载

    Returns:
        EmbeddingService实例
    """
    if config is None:
        config = load_config_from_env()

    return EmbeddingService(config)


# 示例使用
if __name__ == "__main__":
    print("本地embedding服务示例")
    print("="*60)

    try:
        # 创建服务
        service = create_embedding_service()

        # 验证配置
        if service.validate_config():
            # 测试单个文本
            test_text = "这是一个测试文本"
            result = service.get_embedding(test_text)

            if result.error:
                print(f"错误: {result.error}")
            else:
                print(f"成功获取embedding:")
                print(f"  模型: {result.model}")
                print(f"  维度: {result.dimensions}")
                print(f"  Token数: {result.token_count}")
                print(f"  前5个值: {result.embedding[:5]}")

            # 测试批量处理
            print("\n测试批量处理:")
            test_texts = ["第一个测试文本", "第二个测试文本", "第三个测试文本"]
            batch_results = service.get_embeddings_batch(test_texts)

            successful = sum(1 for r in batch_results if not r.error)
            failed = len(batch_results) - successful
            print(f"  批量处理结果: {successful}成功, {failed}失败")

        service.close()

    except Exception as e:
        print(f"初始化失败: {str(e)}")
        print("请确保已安装必要的依赖:")
        print("  pip install torch transformers sentence-transformers")

    print("\n" + "="*60)
    print("示例完成")
    print("="*60)

