#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的PDF解析器示例
基于test.py代码，修复了其中的问题并添加了完整功能
"""

import pdfplumber
import json
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

def main():
    """主函数"""
    # 使用示例PDF文件（需要替换为实际文件）
    pdf_file = "GEA/1.pdf"  # 与test.py中相同的文件名

    # 提取PDF内容
    results = extract_pdf_content(pdf_file)

    # 打印摘要
    print_summary(results)

    # 保存结果到JSON文件（可选）
    if "error" not in results:
        save_results_to_json(results, "pdf_extraction_results.json")

        # 显示前几个文本块作为示例
        print("\n前5个文本块示例:")
        text_blocks = [b for b in results["blocks"] if b["type"] == "text"]
        for i, block in enumerate(text_blocks[:5]):
            print(f"  {i+1}. 页{block['page']}: '{block['content']}'")

if __name__ == "__main__":
    main()

