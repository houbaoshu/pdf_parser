#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF解析器
支持批量处理多个PDF文件，并集成本地embedding功能
使用bge-base-zh-v1.5等本地模型
"""

import pdfplumber
import json
import os
import glob
import sys
from typing import List, Dict, Any, Optional

# 添加embedding服务模块
try:
    from embedding_service import create_embedding_service, EmbeddingConfig, EmbeddingModel
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("警告: embedding_service模块未找到，embedding功能将不可用")
    print("请确保embedding_service.py文件存在，或运行: pip install torch transformers sentence-transformers")


def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    从PDF文件中提取文本、表格和图像内容

    Args:
        pdf_path: PDF文件路径

    Returns:
        包含提取内容的字典
    """
    all_blocks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"正在解析PDF文件: {pdf_path}")
            print(f"总页数: {len(pdf.pages)}")

            for page_idx, page in enumerate(pdf.pages):
                page_number = page_idx + 1

                # 1. 提取文本内容
                words = page.extract_words(use_text_flow=True)
                for word_obj in words:
                    block = {
                        "page": page_number,
                        "type": "text",
                        "bbox": [word_obj["x0"], word_obj["top"],
                                word_obj["x1"], word_obj["bottom"]],
                        "content": word_obj["text"],
                        "metadata": {
                            "x0": word_obj["x0"],
                            "top": word_obj["top"],
                            "x1": word_obj["x1"],
                            "bottom": word_obj["bottom"],
                            "width": word_obj["x1"] - word_obj["x0"],
                            "height": word_obj["bottom"] - word_obj["top"]
                        }
                    }
                    all_blocks.append(block)

                # 2. 提取表格
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    block = {
                        "page": page_number,
                        "type": "table",
                        "table_index": table_idx,
                        "content": table,
                        "metadata": {
                            "rows": len(table),
                            "columns": len(table[0]) if table else 0
                        }
                    }
                    all_blocks.append(block)

                # 3. 提取图像
                for img_idx, image in enumerate(page.images):
                    block = {
                        "page": page_number,
                        "type": "image",
                        "image_index": img_idx,
                        "bbox": [image["x0"], image["top"],
                                image["x1"], image["bottom"]],
                        "metadata": {
                            "x0": image["x0"],
                            "top": image["top"],
                            "x1": image["x1"],
                            "bottom": image["bottom"],
                            "name": image.get("name", ""),
                            "width": image["x1"] - image["x0"],
                            "height": image["bottom"] - image["top"]
                        }
                    }
                    all_blocks.append(block)

                # 显示当前页的提取进度
                print(f"  第{page_number}页: {len(words)}个单词, {len(tables)}个表格, {len(page.images)}个图像")

        # 统计信息
        text_blocks = [b for b in all_blocks if b["type"] == "text"]
        table_blocks = [b for b in all_blocks if b["type"] == "table"]
        image_blocks = [b for b in all_blocks if b["type"] == "image"]

        result = {
            "pdf_path": pdf_path,
            "total_pages": len(pdf.pages) if 'pdf' in locals() else 0,
            "total_blocks": len(all_blocks),
            "statistics": {
                "text_blocks": len(text_blocks),
                "table_blocks": len(table_blocks),
                "image_blocks": len(image_blocks)
            },
            "blocks": all_blocks
        }

        return result

    except FileNotFoundError:
        print(f"错误: 找不到文件 {pdf_path}")
        return {"error": f"File not found: {pdf_path}"}
    except Exception as e:
        print(f"解析PDF时发生错误: {str(e)}")
        return {"error": str(e)}


def save_results_to_json(results: Dict[str, Any], output_path: str):
    """将结果保存为JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果时发生错误: {str(e)}")


def estimate_text_length(text: str) -> int:
    """
    计算文本的字符长度

    Args:
        text: 输入文本

    Returns:
        文本的字符长度
    """
    return len(text) if text else 0


def merge_blocks_to_chunks(blocks: List[Dict[str, Any]], max_chars_per_chunk: int = 800) -> List[Dict[str, Any]]:
    """
    将blocks合并为chunks

    Args:
        blocks: 原始blocks列表
        max_chars_per_chunk: 每个chunk的最大字符数

    Returns:
        合并后的chunks列表
    """
    chunks = []
    current_page = None
    current_text_chunk = []
    current_chunk_chars = 0

    # 按页面和顺序处理blocks
    for block in blocks:
        page = block.get("page", 1)

        # 如果页面变化，完成当前的text chunk
        if current_page is not None and page != current_page and current_text_chunk:
            # 创建text chunk
            chunk_content = " ".join([b["content"] for b in current_text_chunk])
            chunk = {
                "type": "text",
                "page": current_page,
                "content": chunk_content,
                "char_count": current_chunk_chars,
                "block_count": len(current_text_chunk),
                "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                "blocks": current_text_chunk
            }
            chunks.append(chunk)
            current_text_chunk = []
            current_chunk_chars = 0

        current_page = page

        # 处理不同类型的block
        block_type = block.get("type", "text")

        if block_type == "text":
            text_content = block.get("content", "")
            text_chars = estimate_text_length(text_content)

            # 如果当前chunk为空，直接添加
            if not current_text_chunk:
                current_text_chunk.append(block)
                current_chunk_chars = text_chars
            # 如果添加当前block后不超过最大字符数，添加到当前chunk
            elif current_chunk_chars + text_chars <= max_chars_per_chunk:
                current_text_chunk.append(block)
                current_chunk_chars += text_chars
            # 如果超过最大字符数，完成当前chunk并开始新的chunk
            else:
                # 完成当前chunk
                chunk_content = " ".join([b["content"] for b in current_text_chunk])
                chunk = {
                    "type": "text",
                    "page": page,
                    "content": chunk_content,
                    "char_count": current_chunk_chars,
                    "block_count": len(current_text_chunk),
                    "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                    "blocks": current_text_chunk
                }
                chunks.append(chunk)

                # 开始新的chunk
                current_text_chunk = [block]
                current_chunk_chars = text_chars

        elif block_type == "table":
            # 完成当前的text chunk（如果有）
            if current_text_chunk:
                chunk_content = " ".join([b["content"] for b in current_text_chunk])
                chunk = {
                    "type": "text",
                    "page": page,
                    "content": chunk_content,
                    "char_count": current_chunk_chars,
                    "block_count": len(current_text_chunk),
                    "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                    "blocks": current_text_chunk
                }
                chunks.append(chunk)
                current_text_chunk = []
                current_chunk_chars = 0

            # 创建table chunk
            table_content = block.get("content", [])
            # 将表格内容转换为文本
            table_text = ""
            for row in table_content:
                if row:
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    table_text += row_text + "\n"

            table_chars = estimate_text_length(table_text)
            chunk = {
                "type": "table",
                "page": page,
                "table_index": block.get("table_index", 0),
                "content": table_content,
                "text_content": table_text.strip(),
                "char_count": table_chars,
                "block_indices": [blocks.index(block)],
                "blocks": [block],
                "metadata": block.get("metadata", {})
            }
            chunks.append(chunk)

        elif block_type == "image":
            # 完成当前的text chunk（如果有）
            if current_text_chunk:
                chunk_content = " ".join([b["content"] for b in current_text_chunk])
                chunk = {
                    "type": "text",
                    "page": page,
                    "content": chunk_content,
                    "char_count": current_chunk_chars,
                    "block_count": len(current_text_chunk),
                    "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                    "blocks": current_text_chunk
                }
                chunks.append(chunk)
                current_text_chunk = []
                current_chunk_chars = 0

            # 创建image chunk
            chunk = {
                "type": "image",
                "page": page,
                "image_index": block.get("image_index", 0),
                "content": f"图像: {block.get('metadata', {}).get('name', '未命名')}",
                "char_count": 10,  # 图像chunk固定字符数
                "block_indices": [blocks.index(block)],
                "blocks": [block],
                "metadata": block.get("metadata", {})
            }
            chunks.append(chunk)

    # 处理最后一个text chunk（如果有）
    if current_text_chunk:
        chunk_content = " ".join([b["content"] for b in current_text_chunk])
        chunk = {
            "type": "text",
            "page": current_page if current_page else 1,
            "content": chunk_content,
            "char_count": current_chunk_chars,
            "block_count": len(current_text_chunk),
            "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
            "blocks": current_text_chunk
        }
        chunks.append(chunk)

    return chunks


def process_pdf_with_chunks(pdf_path: str, max_chars_per_chunk: int = 800) -> Dict[str, Any]:
    """
    处理PDF文件并生成chunks

    Args:
        pdf_path: PDF文件路径
        max_chars_per_chunk: 每个chunk的最大字符数

    Returns:
        包含chunks的字典
    """
    # 提取原始blocks
    result = extract_pdf_content(pdf_path)

    if "error" in result:
        return result

    # 合并blocks为chunks
    chunks = merge_blocks_to_chunks(result["blocks"], max_chars_per_chunk)

    # 统计信息
    text_chunks = [c for c in chunks if c["type"] == "text"]
    table_chunks = [c for c in chunks if c["type"] == "table"]
    image_chunks = [c for c in chunks if c["type"] == "image"]

    # 更新结果
    result["chunks"] = chunks
    result["chunk_statistics"] = {
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "image_chunks": len(image_chunks),
        "total_chars": sum(c.get("char_count", 0) for c in chunks),
        "avg_chars_per_chunk": sum(c.get("char_count", 0) for c in chunks) / len(chunks) if chunks else 0
    }

    return result


def process_pdf_with_embeddings(pdf_path: str, max_chars_per_chunk: int = 800,
                               embedding_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    处理PDF文件并生成带embedding的chunks

    Args:
        pdf_path: PDF文件路径
        max_chars_per_chunk: 每个chunk的最大字符数
        embedding_config: embedding配置字典

    Returns:
        包含带embedding的chunks的字典
    """
    # 检查embedding功能是否可用
    if not EMBEDDING_AVAILABLE:
        return {"error": "embedding功能不可用，请确保embedding_service.py文件存在"}

    # 处理PDF并生成chunks
    result = process_pdf_with_chunks(pdf_path, max_chars_per_chunk)

    if "error" in result:
        return result

    # 创建embedding服务
    try:
        # 创建配置
        config = EmbeddingConfig()

        # 应用自定义配置
        if embedding_config:
            # 模型配置
            if "model" in embedding_config:
                try:
                    config.model = EmbeddingModel(embedding_config["model"])
                except ValueError:
                    print(f"警告: 未知的模型 {embedding_config['model']}，使用默认模型")

            # 本地模型配置
            if "local_model_name" in embedding_config:
                config.local_model_name = embedding_config["local_model_name"]
            if "device" in embedding_config:
                config.device = embedding_config["device"]
            if "normalize_embeddings" in embedding_config:
                config.normalize_embeddings = embedding_config["normalize_embeddings"]
            if "batch_size" in embedding_config:
                config.batch_size = embedding_config["batch_size"]

        # 创建服务
        service = create_embedding_service(config)

        # 验证配置
        if not service.validate_config():
            service.close()
            return {"error": "embedding配置验证失败，请检查配置"}

        print(f"开始为 {len(result['chunks'])} 个chunks生成embedding...")
        print(f"使用模型: {config.local_model_name}")

        # 为chunks生成embedding
        chunks_with_embeddings = service.get_chunk_embeddings(result["chunks"])

        # 统计embedding结果
        successful_embeddings = 0
        failed_embeddings = 0

        for chunk in chunks_with_embeddings:
            if chunk.get("embedding", {}).get("has_error", True):
                failed_embeddings += 1
            else:
                successful_embeddings += 1

        # 更新结果
        result["chunks"] = chunks_with_embeddings
        result["embedding_statistics"] = {
            "total_chunks": len(chunks_with_embeddings),
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": failed_embeddings,
            "embedding_model": config.local_model_name,
            "embedding_dimensions": service._model.get_sentence_embedding_dimension() if service._model else "未知",
            "embedding_batch_size": config.batch_size
        }

        # 关闭服务
        service.close()

        print(f"embedding完成: {successful_embeddings}成功, {failed_embeddings}失败")

        return result

    except Exception as e:
        return {"error": f"embedding处理失败: {str(e)}"}


def main():
    """主函数"""
    print("="*60)
    print("PDF解析器 - 本地embedding版本")
    print("="*60)
    print("功能:")
    print("1. 解析PDF文件并提取文本、表格、图像")
    print("2. 将提取的内容合并为chunks")
    print("3. 使用bge-base-zh-v1.5模型为chunks生成embedding")
    print("="*60)

    # 测试文件
    test_pdf = "test.pdf"
    if not os.path.exists(test_pdf):
        print(f"测试文件不存在: {test_pdf}")
        print("请创建一个test.pdf文件进行测试")
        return

    # 测试基本PDF解析
    print(f"\n1. 测试PDF解析: {test_pdf}")
    result = extract_pdf_content(test_pdf)

    if "error" in result:
        print(f"解析失败: {result['error']}")
    else:
        print(f"解析成功:")
        print(f"  总页数: {result['total_pages']}")
        print(f"  总块数: {result['total_blocks']}")
        print(f"  文本块: {result['statistics']['text_blocks']}")
        print(f"  表格块: {result['statistics']['table_blocks']}")
        print(f"  图像块: {result['statistics']['image_blocks']}")

        # 保存结果
        save_results_to_json(result, "test_pdf_results.json")

    # 测试chunks处理
    print(f"\n2. 测试chunks处理: {test_pdf}")
    chunk_result = process_pdf_with_chunks(test_pdf, max_chars_per_chunk=800)

    if "error" in chunk_result:
        print(f"chunks处理失败: {chunk_result['error']}")
    else:
        stats = chunk_result.get("chunk_statistics", {})
        print(f"chunks处理成功:")
        print(f"  总chunks数: {stats.get('total_chunks', 0)}")
        print(f"  文本chunks: {stats.get('text_chunks', 0)}")
        print(f"  表格chunks: {stats.get('table_chunks', 0)}")
        print(f"  图像chunks: {stats.get('image_chunks', 0)}")
        print(f"  总字符数: {stats.get('total_chars', 0)}")
        print(f"  平均字符数/chunk: {stats.get('avg_chars_per_chunk', 0):.1f}")

        # 保存结果
        save_results_to_json(chunk_result, "test_pdf_chunks_results.json")

    # 测试embedding功能
    if EMBEDDING_AVAILABLE:
        print(f"\n3. 测试embedding功能: {test_pdf}")
        embedding_result = process_pdf_with_embeddings(test_pdf, max_chars_per_chunk=800)

        if "error" in embedding_result:
            print(f"embedding处理失败: {embedding_result['error']}")
            print("请确保已安装必要的依赖:")
            print("  pip install torch transformers sentence-transformers")
        else:
            emb_stats = embedding_result.get("embedding_statistics", {})
            print(f"embedding处理成功:")
            print(f"  总chunks数: {emb_stats.get('total_chunks', 0)}")
            print(f"  成功embedding: {emb_stats.get('successful_embeddings', 0)}")
            print(f"  失败embedding: {emb_stats.get('failed_embeddings', 0)}")
            print(f"  使用模型: {emb_stats.get('embedding_model', '未知')}")
            print(f"  维度: {emb_stats.get('embedding_dimensions', '未知')}")

            # 保存结果
            save_results_to_json(embedding_result, "test_pdf_embeddings_results.json")

            # 显示前几个带embedding的chunks
            print(f"\n前3个带embedding的chunks示例:")
            for i, chunk in enumerate(embedding_result["chunks"][:3]):
                chunk_type = chunk["type"]
                page = chunk["page"]
                char_count = chunk.get("char_count", 0)

                embedding_info = chunk.get("embedding", {})
                has_embedding = not embedding_info.get("has_error", True)
                embedding_dim = embedding_info.get("dimensions", 0)

                if chunk_type == "text":
                    content_preview = chunk["content"][:80] + "..." if len(chunk["content"]) > 80 else chunk["content"]
                    embedding_status = f"✓ {embedding_dim}维" if has_embedding else "✗ 失败"
                    print(f"  {i+1}. [文本] 页{page}, {char_count}字符, embedding: {embedding_status}")
                    print(f"     内容: '{content_preview}'")
    else:
        print(f"\n3. embedding功能不可用")
        print("请确保:")
        print("  1. embedding_service.py文件存在")
        print("  2. 安装必要的依赖: pip install torch transformers sentence-transformers")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
