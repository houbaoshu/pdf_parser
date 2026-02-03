#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GEA问答Agent的核心功能
"""

import os
import sys
import json
import time

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gea_qa_agent import GEAQAAgent


def test_agent_initialization():
    """测试Agent初始化"""
    print("=" * 80)
    print("测试Agent初始化")
    print("=" * 80)

    agent = GEAQAAgent()

    start_time = time.time()
    success = agent.initialize()
    elapsed_time = time.time() - start_time

    if success:
        print(f"✓ Agent初始化成功")
        print(f"  耗时: {elapsed_time:.2f}秒")
        print(f"  加载chunks: {len(agent.loader.chunks)}")
        print(f"  文本chunks: {len(agent.loader.get_chunks_by_type('text'))}")
        print(f"  表格chunks: {len(agent.loader.get_chunks_by_type('table'))}")
        print(f"  图片chunks: {len(agent.loader.get_chunks_by_type('image'))}")
    else:
        print(f"✗ Agent初始化失败")

    return agent if success else None


def test_text_query(agent):
    """测试文本查询"""
    print("\n" + "=" * 80)
    print("测试文本查询")
    print("=" * 80)

    test_queries = [
        "技术参数",
        "设备特点",
        "操作说明"
    ]

    for query in test_queries:
        print(f"\n查询: '{query}'")
        start_time = time.time()
        result = agent.query(query, query_type="text", top_k=2)
        elapsed_time = time.time() - start_time

        print(f"  耗时: {elapsed_time:.3f}秒")
        print(f"  找到结果: {len(result.search_results)}")

        if result.search_results:
            for i, search_result in enumerate(result.search_results[:2]):
                chunk = search_result.chunk
                print(f"  {i+1}. [{chunk.type}] 相似度: {search_result.similarity:.4f}")
                print(f"     来源: {os.path.basename(chunk.source_file)}")
                print(f"     页码: {chunk.page}")
                content_preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
                print(f"     内容: {content_preview}")


def test_image_query(agent):
    """测试图片查询"""
    print("\n" + "=" * 80)
    print("测试图片查询")
    print("=" * 80)

    test_queries = [
        "图片",
        "图表",
        "示意图"
    ]

    for query in test_queries:
        print(f"\n查询: '{query}'")
        start_time = time.time()
        result = agent.query(query, query_type="image", top_k=2)
        elapsed_time = time.time() - start_time

        print(f"  耗时: {elapsed_time:.3f}秒")
        print(f"  找到结果: {len(result.search_results)}")

        if result.search_results:
            for i, search_result in enumerate(result.search_results[:2]):
                chunk = search_result.chunk
                print(f"  {i+1}. [{chunk.type}] 相似度: {search_result.similarity:.4f}")
                print(f"     来源: {os.path.basename(chunk.source_file)}")
                print(f"     页码: {chunk.page}")
                print(f"     内容: {chunk.content}")


def test_image_search(agent):
    """测试图片搜索"""
    print("\n" + "=" * 80)
    print("测试图片搜索")
    print("=" * 80)

    keywords = ["图", "表", "image"]

    for keyword in keywords:
        print(f"\n搜索图片关键词: '{keyword}'")
        start_time = time.time()
        results = agent.search_images(keyword, top_k=3)
        elapsed_time = time.time() - start_time

        print(f"  耗时: {elapsed_time:.3f}秒")
        print(f"  找到图片: {len(results)}")

        if results:
            for i, result in enumerate(results):
                chunk = result.chunk
                print(f"  {i+1}. 相似度: {result.similarity:.4f}")
                print(f"     来源: {os.path.basename(chunk.source_file)}")
                print(f"     页码: {chunk.page}")
                print(f"     内容: {chunk.content}")
                if chunk.metadata:
                    print(f"     元数据: {chunk.metadata}")


def test_answer_generation(agent):
    """测试答案生成"""
    print("\n" + "=" * 80)
    print("测试答案生成")
    print("=" * 80)

    questions = [
        "GEA设备的主要功能是什么？",
        "设备的技术参数有哪些？",
        "如何操作这个设备？"
    ]

    for question in questions:
        print(f"\n问题: '{question}'")
        start_time = time.time()
        answer = agent.get_answer_with_context(question, include_context=True)
        elapsed_time = time.time() - start_time

        print(f"  耗时: {elapsed_time:.3f}秒")
        print(f"  找到答案: {answer['has_answer']}")
        print(f"  置信度: {answer.get('confidence', 0):.4f}")

        if answer['has_answer']:
            print(f"  答案预览: {answer['answer'][:150]}...")
            if answer['sources']:
                print("  来源:")
                for source in answer['sources']:
                    print(f"    - {os.path.basename(source['source_file'])} (页{source['page']})")


def test_chunk_loading():
    """测试chunk加载"""
    print("=" * 80)
    print("测试chunk加载")
    print("=" * 80)

    # 检查embedding文件
    embeddings_dir = "embeddings/GEA"
    if not os.path.exists(embeddings_dir):
        print(f"✗ embedding目录不存在: {embeddings_dir}")
        return

    # 列出embedding文件
    embedding_files = []
    for file in os.listdir(embeddings_dir):
        if file.endswith("_embeddings.json") and file != "processing_summary.json":
            file_path = os.path.join(embeddings_dir, file)
            file_size = os.path.getsize(file_path)
            embedding_files.append((file, file_size))

    print(f"找到 {len(embedding_files)} 个embedding文件:")
    for file, size in embedding_files:
        size_mb = size / (1024 * 1024)
        print(f"  - {file}: {size_mb:.2f} MB")

    # 检查processing_summary.json
    summary_file = os.path.join(embeddings_dir, "processing_summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            print(f"\n处理摘要:")
            print(f"  总文件数: {summary.get('summary', {}).get('total_files', 0)}")
            print(f"  总chunks数: {summary.get('summary', {}).get('total_chunks', 0)}")
            print(f"  成功embedding: {summary.get('summary', {}).get('total_successful_embeddings', 0)}")
        except Exception as e:
            print(f"  读取摘要文件失败: {str(e)}")


def main():
    """主测试函数"""
    print("GEA问答Agent功能测试")
    print("=" * 80)

    # 测试chunk加载
    test_chunk_loading()

    # 测试Agent初始化
    agent = test_agent_initialization()

    if not agent:
        print("\n" + "=" * 80)
        print("Agent初始化失败，无法继续测试")
        print("=" * 80)
        return

    # 运行其他测试
    test_text_query(agent)
    test_image_query(agent)
    test_image_search(agent)
    test_answer_generation(agent)

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

