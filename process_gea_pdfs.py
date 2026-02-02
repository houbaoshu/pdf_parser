#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理GEA文件夹下的所有PDF文件，生成embedding并保存到embeddings文件夹
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径，以便导入pdf_parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pdf_parser import process_pdf_with_embeddings
    from embedding_service import EmbeddingConfig, EmbeddingModel
    EMBEDDING_AVAILABLE = True
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    EMBEDDING_AVAILABLE = False


def get_gea_pdf_files() -> List[str]:
    """
    获取GEA文件夹下的所有PDF文件

    Returns:
        PDF文件路径列表
    """
    pdf_files = []

    # 查找GEA文件夹下的PDF文件
    for root, dirs, files in os.walk("."):
        if "GEA" in root:
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    pdf_files.append(pdf_path)

    # 如果没有找到，使用硬编码的路径
    if not pdf_files:
        logger.warning("未找到PDF文件，使用硬编码路径")
        pdf_files = [
            "GEA/1.pdf",
            "GEA/2.pdf",
            "GEA/3.pdf",
            "GEA/4.pdf",
            "GEA/5.pdf"
        ]

    # 过滤掉不存在的文件
    existing_files = []
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            existing_files.append(pdf_file)
        else:
            logger.warning(f"文件不存在: {pdf_file}")

    return existing_files


def create_output_directory() -> str:
    """
    创建输出目录

    Returns:
        输出目录路径
    """
    # 创建embeddings目录
    output_dir = "embeddings"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    # 创建GEA子目录
    gea_output_dir = os.path.join(output_dir, "GEA")
    if not os.path.exists(gea_output_dir):
        os.makedirs(gea_output_dir)
        logger.info(f"创建GEA输出目录: {gea_output_dir}")

    return gea_output_dir


def process_pdf_file(pdf_path: str, output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个PDF文件

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        config: embedding配置

    Returns:
        处理结果
    """
    logger.info(f"开始处理PDF文件: {pdf_path}")

    try:
        # 获取文件名（不含扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # 处理PDF并生成embedding
        start_time = time.time()
        result = process_pdf_with_embeddings(
            pdf_path=pdf_path,
            max_chars_per_chunk=800,
            embedding_config=config
        )
        processing_time = time.time() - start_time

        # 检查是否有错误
        if "error" in result:
            logger.error(f"处理PDF文件失败: {result['error']}")
            return {
                "pdf_path": pdf_path,
                "status": "failed",
                "error": result["error"],
                "processing_time": processing_time
            }

        # 保存结果到JSON文件
        output_file = os.path.join(output_dir, f"{pdf_name}_embeddings.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 提取统计信息
        stats = result.get("chunk_statistics", {})
        emb_stats = result.get("embedding_statistics", {})

        # 记录成功信息
        logger.info(f"PDF文件处理完成: {pdf_path}")
        logger.info(f"  总chunks数: {stats.get('total_chunks', 0)}")
        logger.info(f"  成功embedding: {emb_stats.get('successful_embeddings', 0)}")
        logger.info(f"  失败embedding: {emb_stats.get('failed_embeddings', 0)}")
        logger.info(f"  处理时间: {processing_time:.2f}秒")
        logger.info(f"  结果保存到: {output_file}")

        return {
            "pdf_path": pdf_path,
            "status": "success",
            "output_file": output_file,
            "total_chunks": stats.get("total_chunks", 0),
            "successful_embeddings": emb_stats.get("successful_embeddings", 0),
            "failed_embeddings": emb_stats.get("failed_embeddings", 0),
            "processing_time": processing_time,
            "embedding_model": emb_stats.get("embedding_model", "未知")
        }

    except Exception as e:
        logger.error(f"处理PDF文件时发生异常: {str(e)}")
        return {
            "pdf_path": pdf_path,
            "status": "failed",
            "error": str(e)
        }


def generate_summary_report(results: List[Dict[str, Any]], output_dir: str):
    """
    生成处理摘要报告

    Args:
        results: 处理结果列表
        output_dir: 输出目录
    """
    # 统计信息
    total_files = len(results)
    successful_files = sum(1 for r in results if r.get("status") == "success")
    failed_files = total_files - successful_files

    total_chunks = sum(r.get("total_chunks", 0) for r in results if r.get("status") == "success")
    total_successful_embeddings = sum(r.get("successful_embeddings", 0) for r in results if r.get("status") == "success")
    total_failed_embeddings = sum(r.get("failed_embeddings", 0) for r in results if r.get("status") == "success")
    total_processing_time = sum(r.get("processing_time", 0) for r in results if r.get("status") == "success")

    # 创建摘要报告
    summary = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "total_successful_embeddings": total_successful_embeddings,
            "total_failed_embeddings": total_failed_embeddings,
            "total_processing_time": total_processing_time,
            "avg_processing_time_per_file": total_processing_time / successful_files if successful_files > 0 else 0
        },
        "file_results": results
    }

    # 保存摘要报告
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"摘要报告已保存到: {summary_file}")

    # 打印摘要
    print("\n" + "="*60)
    print("处理摘要报告")
    print("="*60)
    print(f"总文件数: {total_files}")
    print(f"成功处理: {successful_files}")
    print(f"失败处理: {failed_files}")
    print(f"总chunks数: {total_chunks}")
    print(f"总成功embedding: {total_successful_embeddings}")
    print(f"总失败embedding: {total_failed_embeddings}")
    print(f"总处理时间: {total_processing_time:.2f}秒")
    if successful_files > 0:
        print(f"平均处理时间/文件: {total_processing_time/successful_files:.2f}秒")
    print("="*60)

    # 打印失败的文件
    if failed_files > 0:
        print("\n失败的文件:")
        for result in results:
            if result.get("status") == "failed":
                print(f"  - {result['pdf_path']}: {result.get('error', '未知错误')}")


def main():
    """主函数"""
    print("="*60)
    print("GEA PDF文件批量处理工具")
    print("="*60)

    # 检查embedding功能是否可用
    if not EMBEDDING_AVAILABLE:
        print("错误: embedding功能不可用")
        print("请确保:")
        print("  1. embedding_service.py文件存在")
        print("  2. 安装必要的依赖: pip install torch transformers sentence-transformers")
        return

    # 获取PDF文件
    print("\n1. 查找GEA文件夹下的PDF文件...")
    pdf_files = get_gea_pdf_files()

    if not pdf_files:
        print("错误: 未找到PDF文件")
        return

    print(f"找到 {len(pdf_files)} 个PDF文件:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")

    # 创建输出目录
    print("\n2. 创建输出目录...")
    output_dir = create_output_directory()
    print(f"输出目录: {output_dir}")

    # 配置embedding参数
    print("\n3. 配置embedding参数...")
    embedding_config = {
        "model": "BAAI/bge-base-zh-v1.5",
        "local_model_name": "BAAI/bge-base-zh-v1.5",
        "device": "cpu",  # 使用CPU，如果需要GPU可改为"cuda"
        "normalize_embeddings": True,
        "batch_size": 100
    }

    print(f"使用模型: {embedding_config['local_model_name']}")
    print(f"使用设备: {embedding_config['device']}")
    print(f"批量大小: {embedding_config['batch_size']}")

    # 处理所有PDF文件
    print("\n4. 开始处理PDF文件...")
    print("="*60)

    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n处理文件 {i}/{len(pdf_files)}: {pdf_file}")
        result = process_pdf_file(pdf_file, output_dir, embedding_config)
        results.append(result)

        # 添加延迟，避免资源过度使用
        if i < len(pdf_files):
            time.sleep(1)

    # 生成摘要报告
    print("\n5. 生成处理摘要...")
    generate_summary_report(results, output_dir)

    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)


if __name__ == "__main__":
    main()

