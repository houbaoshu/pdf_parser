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
from typing import List, Dict, Any

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

if __name__ == "__main__":
    main()

