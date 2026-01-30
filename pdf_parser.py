#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF解析器
支持批量处理多个PDF文件
"""

import pdfplumber
import json
import os
import glob
import re
from typing import List, Dict, Any, Tuple

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

def print_summary(results: Dict[str, Any]):
    """打印提取结果的摘要"""
    if "error" in results:
        print(f"错误: {results['error']}")
        return

    print("\n" + "="*50)
    print("PDF解析结果摘要")
    print("="*50)
    print(f"PDF文件: {results['pdf_path']}")
    print(f"总页数: {results['total_pages']}")
    print(f"总块数: {results['total_blocks']}")
    print(f"文本块: {results['statistics']['text_blocks']}")
    print(f"表格块: {results['statistics']['table_blocks']}")
    print(f"图像块: {results['statistics']['image_blocks']}")
    print("="*50)

def batch_process_pdfs(pdf_dir: str, pattern: str = "*.pdf") -> Dict[str, Any]:
    """
    批量处理PDF文件

    Args:
        pdf_dir: PDF文件目录
        pattern: 文件匹配模式

    Returns:
        包含所有处理结果的字典
    """
    pdf_files = glob.glob(os.path.join(pdf_dir, pattern))
    pdf_files.sort()  # 按文件名排序

    if not pdf_files:
        print(f"在目录 {pdf_dir} 中没有找到匹配 {pattern} 的PDF文件")
        return {"error": f"No PDF files found in {pdf_dir} matching {pattern}"}

    print(f"找到 {len(pdf_files)} 个PDF文件:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {os.path.basename(pdf_file)}")

    all_results = {}
    total_summary = {
        "total_files": len(pdf_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_pages": 0,
        "total_blocks": 0,
        "total_text_blocks": 0,
        "total_table_blocks": 0,
        "total_image_blocks": 0
    }

    print("\n" + "="*60)
    print("开始批量处理PDF文件")
    print("="*60)

    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        print(f"\n处理文件: {filename}")

        result = extract_pdf_content(pdf_file)

        if "error" in result:
            print(f"  ✗ 处理失败: {result['error']}")
            all_results[filename] = {"error": result["error"]}
            total_summary["failed_files"] += 1
        else:
            print(f"  ✓ 处理成功")
            all_results[filename] = result
            total_summary["processed_files"] += 1
            total_summary["total_pages"] += result["total_pages"]
            total_summary["total_blocks"] += result["total_blocks"]
            total_summary["total_text_blocks"] += result["statistics"]["text_blocks"]
            total_summary["total_table_blocks"] += result["statistics"]["table_blocks"]
            total_summary["total_image_blocks"] += result["statistics"]["image_blocks"]

    print("\n" + "="*60)
    print("批量处理完成")
    print("="*60)

    return {
        "batch_summary": total_summary,
        "results": all_results
    }

def print_batch_summary(batch_results: Dict[str, Any]):
    """打印批量处理结果的摘要"""
    if "error" in batch_results:
        print(f"批量处理错误: {batch_results['error']}")
        return

    summary = batch_results["batch_summary"]

    print("\n" + "="*60)
    print("批量处理结果摘要")
    print("="*60)
    print(f"总文件数: {summary['total_files']}")
    print(f"成功处理: {summary['processed_files']}")
    print(f"处理失败: {summary['failed_files']}")
    print(f"总页数: {summary['total_pages']}")
    print(f"总块数: {summary['total_blocks']}")
    print(f"文本块总数: {summary['total_text_blocks']}")
    print(f"表格块总数: {summary['total_table_blocks']}")
    print(f"图像块总数: {summary['total_image_blocks']}")
    print("="*60)

    # 打印每个文件的简要信息
    if "results" in batch_results:
        print("\n各文件详情:")
        for filename, result in batch_results["results"].items():
            if "error" in result:
                print(f"  {filename}: 失败 - {result['error']}")
            else:
                print(f"  {filename}: {result['total_pages']}页, {result['total_blocks']}块 "
                      f"(文本:{result['statistics']['text_blocks']}, "
                      f"表格:{result['statistics']['table_blocks']}, "
                      f"图像:{result['statistics']['image_blocks']})")

def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量

    Args:
        text: 输入文本

    Returns:
        估算的token数量
    """
    if not text:
        return 0

    # 简单估算方法：
    # 1. 对于英文：一个单词 ≈ 1.3个tokens
    # 2. 对于中文：一个汉字 ≈ 2个tokens
    # 3. 对于标点符号和数字：每个字符 ≈ 0.5个tokens

    # 统计中文字符
    chinese_chars = len(re.findall(r'[一-鿿]', text))

    # 统计英文字符（排除中文字符）
    english_text = re.sub(r'[一-鿿]', '', text)
    # 简单的单词分割：按空格和非字母数字字符分割
    english_words = re.findall(r'\b\w+\b', english_text)

    # 统计其他字符（标点、数字等）
    other_chars = len(text) - chinese_chars - sum(len(word) for word in english_words)

    # 计算token估算
    tokens = chinese_chars * 2 + len(english_words) * 1.3 + other_chars * 0.5

    return int(tokens + 0.5)  # 四舍五入到最接近的整数

def merge_blocks_to_chunks(blocks: List[Dict[str, Any]], max_tokens_per_chunk: int = 400) -> List[Dict[str, Any]]:
    """
    将blocks合并为chunks

    Args:
        blocks: 原始blocks列表
        max_tokens_per_chunk: 每个chunk的最大token数

    Returns:
        合并后的chunks列表
    """
    chunks = []
    current_page = None
    current_text_chunk = []
    current_chunk_tokens = 0

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
                "token_count": current_chunk_tokens,
                "block_count": len(current_text_chunk),
                "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                "blocks": current_text_chunk
            }
            chunks.append(chunk)
            current_text_chunk = []
            current_chunk_tokens = 0

        current_page = page

        # 处理不同类型的block
        block_type = block.get("type", "text")

        if block_type == "text":
            text_content = block.get("content", "")
            text_tokens = estimate_tokens(text_content)

            # 如果当前chunk为空，直接添加
            if not current_text_chunk:
                current_text_chunk.append(block)
                current_chunk_tokens = text_tokens
            # 如果添加当前block后不超过最大token数，添加到当前chunk
            elif current_chunk_tokens + text_tokens <= max_tokens_per_chunk:
                current_text_chunk.append(block)
                current_chunk_tokens += text_tokens
            # 如果超过最大token数，完成当前chunk并开始新的chunk
            else:
                # 完成当前chunk
                chunk_content = " ".join([b["content"] for b in current_text_chunk])
                chunk = {
                    "type": "text",
                    "page": page,
                    "content": chunk_content,
                    "token_count": current_chunk_tokens,
                    "block_count": len(current_text_chunk),
                    "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                    "blocks": current_text_chunk
                }
                chunks.append(chunk)

                # 开始新的chunk
                current_text_chunk = [block]
                current_chunk_tokens = text_tokens

        elif block_type == "table":
            # 完成当前的text chunk（如果有）
            if current_text_chunk:
                chunk_content = " ".join([b["content"] for b in current_text_chunk])
                chunk = {
                    "type": "text",
                    "page": page,
                    "content": chunk_content,
                    "token_count": current_chunk_tokens,
                    "block_count": len(current_text_chunk),
                    "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                    "blocks": current_text_chunk
                }
                chunks.append(chunk)
                current_text_chunk = []
                current_chunk_tokens = 0

            # 创建table chunk
            table_content = block.get("content", [])
            # 将表格内容转换为文本
            table_text = ""
            for row in table_content:
                if row:
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    table_text += row_text + "\n"

            table_tokens = estimate_tokens(table_text)
            chunk = {
                "type": "table",
                "page": page,
                "table_index": block.get("table_index", 0),
                "content": table_content,
                "text_content": table_text.strip(),
                "token_count": table_tokens,
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
                    "token_count": current_chunk_tokens,
                    "block_count": len(current_text_chunk),
                    "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
                    "blocks": current_text_chunk
                }
                chunks.append(chunk)
                current_text_chunk = []
                current_chunk_tokens = 0

            # 创建image chunk
            chunk = {
                "type": "image",
                "page": page,
                "image_index": block.get("image_index", 0),
                "content": f"图像: {block.get('metadata', {}).get('name', '未命名')}",
                "token_count": 10,  # 图像chunk固定token数
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
            "token_count": current_chunk_tokens,
            "block_count": len(current_text_chunk),
            "block_indices": [i for i, b in enumerate(blocks) if b in current_text_chunk],
            "blocks": current_text_chunk
        }
        chunks.append(chunk)

    return chunks

def process_pdf_with_chunks(pdf_path: str, max_tokens_per_chunk: int = 400) -> Dict[str, Any]:
    """
    处理PDF文件并生成chunks

    Args:
        pdf_path: PDF文件路径
        max_tokens_per_chunk: 每个chunk的最大token数

    Returns:
        包含chunks的字典
    """
    # 提取原始blocks
    result = extract_pdf_content(pdf_path)

    if "error" in result:
        return result

    # 合并blocks为chunks
    chunks = merge_blocks_to_chunks(result["blocks"], max_tokens_per_chunk)

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
        "total_tokens": sum(c.get("token_count", 0) for c in chunks),
        "avg_tokens_per_chunk": sum(c.get("token_count", 0) for c in chunks) / len(chunks) if chunks else 0
    }

    return result

def print_chunk_summary(results: Dict[str, Any]):
    """打印chunk处理结果的摘要"""
    if "error" in results:
        print(f"错误: {results['error']}")
        return

    if "chunks" not in results:
        print("未找到chunks信息")
        return

    stats = results.get("chunk_statistics", {})

    print("\n" + "="*60)
    print("Chunk处理结果摘要")
    print("="*60)
    print(f"PDF文件: {results['pdf_path']}")
    print(f"总页数: {results['total_pages']}")
    print(f"原始块数: {results['total_blocks']}")
    print(f"合并后chunk数: {stats.get('total_chunks', 0)}")
    print(f"文本chunks: {stats.get('text_chunks', 0)}")
    print(f"表格chunks: {stats.get('table_chunks', 0)}")
    print(f"图像chunks: {stats.get('image_chunks', 0)}")
    print(f"总tokens: {stats.get('total_tokens', 0)}")
    print(f"平均tokens/chunk: {stats.get('avg_tokens_per_chunk', 0):.1f}")
    print("="*60)

    # 显示前几个chunks作为示例
    print("\n前5个chunks示例:")
    for i, chunk in enumerate(results["chunks"][:5]):
        chunk_type = chunk["type"]
        page = chunk["page"]
        token_count = chunk.get("token_count", 0)

        if chunk_type == "text":
            content_preview = chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"]
            print(f"  {i+1}. [文本] 页{page}, {token_count}tokens: '{content_preview}'")
        elif chunk_type == "table":
            rows = len(chunk["content"]) if chunk["content"] else 0
            cols = len(chunk["content"][0]) if chunk["content"] and chunk["content"][0] else 0
            print(f"  {i+1}. [表格] 页{page}, {rows}行×{cols}列, {token_count}tokens")
        elif chunk_type == "image":
            image_name = chunk.get("metadata", {}).get("name", "未命名")
            print(f"  {i+1}. [图像] 页{page}, {image_name}, {token_count}tokens")

def main():
    """主函数"""
    print("="*60)
    print("PDF批量解析器 - 处理GEA文件夹中的所有PDF文件")
    print("="*60)

    # 批量处理GEA文件夹中的所有PDF文件
    batch_results = batch_process_pdfs("GEA", "*.pdf")

    # 打印批量处理摘要
    print_batch_summary(batch_results)

    # 保存批量处理结果到JSON文件
    if "error" not in batch_results:
        save_results_to_json(batch_results, "gea_pdf_batch_results.json")

        # 为每个文件单独保存结果
        if "results" in batch_results:
            for filename, result in batch_results["results"].items():
                if "error" not in result:
                    # 为每个文件创建单独的JSON文件
                    output_filename = f"gea_{filename.replace('.pdf', '')}_results.json"
                    save_results_to_json(result, output_filename)

                    # 显示每个文件的前几个文本块作为示例
                    print(f"\n{filename} 前3个文本块示例:")
                    text_blocks = [b for b in result["blocks"] if b["type"] == "text"]
                    for i, block in enumerate(text_blocks[:3]):
                        print(f"  {i+1}. 页{block['page']}: '{block['content']}'")

    print("\n" + "="*60)
    print("Chunk合并处理 - 测试1.pdf")
    print("="*60)

    # 测试chunk合并功能
    test_pdf = "GEA/1.pdf"
    if os.path.exists(test_pdf):
        chunk_result = process_pdf_with_chunks(test_pdf, max_tokens_per_chunk=400)
        print_chunk_summary(chunk_result)

        # 保存chunk结果
        if "error" not in chunk_result:
            save_results_to_json(chunk_result, "gea_1_chunks_results.json")
            print(f"\nChunk结果已保存到: gea_1_chunks_results.json")
    else:
        print(f"测试文件不存在: {test_pdf}")

if __name__ == "__main__":
    main()

