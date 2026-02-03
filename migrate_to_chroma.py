#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON格式的embeddings迁移到Chroma向量数据库
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaMigrator:
    """Chroma迁移器"""

    def __init__(self,
                 embeddings_dir: str = "embeddings/GEA",
                 chroma_db_path: str = "chroma_db"):
        """
        初始化迁移器

        Args:
            embeddings_dir: JSON embeddings目录
            chroma_db_path: Chroma数据库路径
        """
        self.embeddings_dir = embeddings_dir
        self.chroma_db_path = chroma_db_path

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.collection = None

    def create_collection(self, collection_name: str = "gea_documents"):
        """
        创建或获取Chroma集合

        Args:
            collection_name: 集合名称
        """
        try:
            # 尝试获取现有集合
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"找到现有集合: {collection_name}")

            # 询问是否重置
            logger.warning(f"集合已存在，包含 {self.collection.count()} 个文档")
            response = input("是否删除并重新创建？(y/n): ")

            if response.lower() == 'y':
                self.client.delete_collection(name=collection_name)
                logger.info(f"已删除集合: {collection_name}")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "GEA文档embeddings"}
                )
                logger.info(f"创建新集合: {collection_name}")
        except Exception:
            # 集合不存在，创建新的
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "GEA文档embeddings"}
            )
            logger.info(f"创建新集合: {collection_name}")

    def load_json_embeddings(self) -> List[Dict[str, Any]]:
        """
        加载JSON格式的embeddings

        Returns:
            chunks列表
        """
        if not os.path.exists(self.embeddings_dir):
            logger.error(f"embedding目录不存在: {self.embeddings_dir}")
            return []

        # 查找所有embedding文件
        embedding_files = []
        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_embeddings.json") and file != "processing_summary.json":
                embedding_files.append(os.path.join(self.embeddings_dir, file))

        if not embedding_files:
            logger.error(f"在 {self.embeddings_dir} 中未找到embedding文件")
            return []

        logger.info(f"找到 {len(embedding_files)} 个embedding文件")

        # 加载所有chunks
        all_chunks = []
        for file_path in embedding_files:
            try:
                logger.info(f"加载文件: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取chunks
                chunks_data = data.get("chunks", [])
                source_file = data.get("pdf_path", "")

                for i, chunk_data in enumerate(chunks_data):
                    # 提取embedding向量
                    embedding = chunk_data.get("embedding", {}).get("vector", [])

                    # 只保留有效的embedding
                    if not embedding or len(embedding) == 0:
                        continue

                    # 生成唯一ID
                    chunk_id = hashlib.md5(f"{source_file}_{i}".encode()).hexdigest()[:16]

                    # 处理content - 确保是字符串
                    content = chunk_data.get("content", "")
                    if isinstance(content, list):
                        # 表格类型的content是列表，转换为字符串
                        content = json.dumps(content, ensure_ascii=False)
                    elif not isinstance(content, str):
                        content = str(content)

                    chunk_info = {
                        "id": chunk_id,
                        "content": content,
                        "embedding": embedding,
                        "metadata": {
                            "type": chunk_data.get("type", "text"),
                            "page": chunk_data.get("page", 1),
                            "source_file": source_file,
                            "char_count": chunk_data.get("char_count", 0),
                            "block_indices": json.dumps(chunk_data.get("block_indices", [])),
                            # Chroma metadata必须是基本类型，不能是list/dict
                            "has_blocks": len(chunk_data.get("blocks", [])) > 0
                        }
                    }

                    all_chunks.append(chunk_info)

                logger.info(f"  加载 {len(chunks_data)} chunks (有效: {len([c for c in chunks_data if c.get('embedding', {}).get('vector')])})")

            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {str(e)}")
                continue

        logger.info(f"总共加载 {len(all_chunks)} 个有效chunks")
        return all_chunks

    def migrate_to_chroma(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        将chunks迁移到Chroma

        Args:
            chunks: chunks列表
            batch_size: 批处理大小
        """
        if not self.collection:
            logger.error("集合未创建，请先调用create_collection()")
            return

        if not chunks:
            logger.error("没有chunks可以迁移")
            return

        logger.info(f"开始迁移 {len(chunks)} 个chunks到Chroma")

        # 批量添加
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(chunks), batch_size),
                     total=total_batches,
                     desc="迁移进度"):
            batch = chunks[i:i + batch_size]

            # 准备批次数据
            ids = [chunk["id"] for chunk in batch]
            embeddings = [chunk["embedding"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]

            try:
                # 添加到Chroma
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                logger.error(f"批次 {i//batch_size + 1} 迁移失败: {str(e)}")
                continue

        # 验证迁移结果
        final_count = self.collection.count()
        logger.info(f"迁移完成！Chroma中共有 {final_count} 个文档")

        # 显示统计信息
        self._show_statistics()

    def _show_statistics(self):
        """显示统计信息"""
        if not self.collection:
            return

        try:
            # 按类型统计
            text_count = self.collection.count()  # 总数

            # 获取样本来统计类型
            sample = self.collection.get(limit=1000)
            if sample and "metadatas" in sample:
                type_counts = {}
                for meta in sample["metadatas"]:
                    chunk_type = meta.get("type", "unknown")
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

                logger.info("文档类型分布（样本）:")
                for chunk_type, count in type_counts.items():
                    logger.info(f"  {chunk_type}: {count}")
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")

    def verify_migration(self, sample_size: int = 5):
        """
        验证迁移结果

        Args:
            sample_size: 抽样验证数量
        """
        if not self.collection:
            logger.error("集合未创建")
            return

        logger.info(f"\n验证迁移结果（抽样 {sample_size} 个）:")
        logger.info("=" * 80)

        try:
            # 获取样本
            results = self.collection.get(
                limit=sample_size,
                include=["documents", "metadatas", "embeddings"]
            )

            if not results or not results["ids"]:
                logger.warning("未找到任何文档")
                return

            for i, (doc_id, document, metadata, embedding) in enumerate(
                zip(results["ids"], results["documents"], results["metadatas"], results["embeddings"])
            ):
                logger.info(f"\n文档 {i+1}:")
                logger.info(f"  ID: {doc_id}")
                logger.info(f"  类型: {metadata.get('type')}")
                logger.info(f"  来源: {metadata.get('source_file')}")
                logger.info(f"  页码: {metadata.get('page')}")
                logger.info(f"  字符数: {metadata.get('char_count')}")
                logger.info(f"  Embedding维度: {len(embedding)}")
                content_preview = document[:100] + "..." if len(document) > 100 else document
                logger.info(f"  内容预览: {content_preview}")

        except Exception as e:
            logger.error(f"验证失败: {str(e)}")


def main():
    """主函数"""
    print("=" * 80)
    print("JSON Embeddings → Chroma 迁移工具")
    print("=" * 80)

    # 创建迁移器
    migrator = ChromaMigrator(
        embeddings_dir="embeddings/GEA",
        chroma_db_path="chroma_db"
    )

    # 创建集合
    migrator.create_collection(collection_name="gea_documents")

    # 加载JSON embeddings
    print("\n步骤1: 加载JSON embeddings...")
    chunks = migrator.load_json_embeddings()

    if not chunks:
        print("没有找到有效的chunks，退出")
        return

    # 迁移到Chroma
    print("\n步骤2: 迁移到Chroma...")
    migrator.migrate_to_chroma(chunks, batch_size=100)

    # 验证迁移
    print("\n步骤3: 验证迁移结果...")
    migrator.verify_migration(sample_size=5)

    print("\n" + "=" * 80)
    print("迁移完成！")
    print(f"Chroma数据库位置: {migrator.chroma_db_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
