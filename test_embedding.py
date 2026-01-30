#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试本地embedding功能
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_service import EmbeddingConfig, EmbeddingModel, create_embedding_service


def test_embedding_service_structure():
    """测试embedding服务结构"""
    print("="*60)
    print("测试本地embedding服务结构")
    print("="*60)

    # 测试配置类
    config = EmbeddingConfig()
    print(f"1. 默认配置:")
    print(f"   模型: {config.model.value}")
    print(f"   本地模型名称: {config.local_model_name}")
    print(f"   设备: {config.device}")
    print(f"   超时时间: {config.timeout}秒")
    print(f"   批量大小: {config.batch_size}")
    print(f"   归一化embedding: {config.normalize_embeddings}")

    # 测试模型枚举
    print(f"\n2. 支持的本地模型:")
    for model in EmbeddingModel:
        print(f"   - {model.value}")

    # 测试服务创建
    try:
        service = create_embedding_service(config)
        print(f"\n3. 服务创建: 成功")
        print(f"   embedding功能可用: 是")

        # 测试配置验证
        print(f"\n4. 配置验证:")
        if service.validate_config():
            print(f"   配置验证: 成功")
        else:
            print(f"   配置验证: 失败")

        service.close()
        print(f"\n5. 服务关闭: 成功")

    except Exception as e:
        print(f"\n3. 服务创建: 失败")
        print(f"   错误: {str(e)}")
        print(f"   embedding功能可用: 否")
        print(f"   请确保已安装必要的依赖:")
        print(f"   pip install torch transformers sentence-transformers")

    print("\n" + "="*60)
    print("结构测试完成")
    print("="*60)


def test_pdf_parser_integration():
    """测试PDF解析器集成"""
    print("\n" + "="*60)
    print("测试PDF解析器集成")
    print("="*60)

    try:
        import pdf_parser

        print("1. PDF解析器导入: 成功")

        # 检查embedding可用性
        if pdf_parser.EMBEDDING_AVAILABLE:
            print("2. embedding功能集成: 成功")
            print("   支持的功能:")
            print("   - process_pdf_with_embeddings()")
            print("   - 使用bge-base-zh-v1.5等本地模型")
        else:
            print("2. embedding功能集成: 失败")
            print("   请确保embedding_service.py文件存在")

        # 测试PDF文件存在性
        test_pdf = "test.pdf"
        if os.path.exists(test_pdf):
            print(f"\n3. 测试PDF文件: 存在 ({test_pdf})")
            print(f"   文件大小: {os.path.getsize(test_pdf)} 字节")
        else:
            print(f"\n3. 测试PDF文件: 不存在 ({test_pdf})")
            print("   请创建一个test.pdf文件进行测试")

    except ImportError as e:
        print(f"1. PDF解析器导入: 失败")
        print(f"   错误: {str(e)}")
    except Exception as e:
        print(f"1. PDF解析器导入: 失败")
        print(f"   错误: {str(e)}")

    print("\n" + "="*60)
    print("集成测试完成")
    print("="*60)


def test_local_embedding_functionality():
    """测试本地embedding功能"""
    print("\n" + "="*60)
    print("测试本地embedding功能")
    print("="*60)

    try:
        # 创建配置
        config = EmbeddingConfig()

        # 创建服务
        service = create_embedding_service(config)

        print("1. 服务初始化: 成功")
        print(f"   使用模型: {config.local_model_name}")
        print(f"   使用设备: {config.device}")

        # 测试单个文本embedding
        print("\n2. 测试单个文本embedding:")
        test_text = "这是一个测试文本，用于验证本地embedding功能"
        result = service.get_embedding(test_text)

        if result.error:
            print(f"   失败: {result.error}")
        else:
            print(f"   成功: {result.dimensions}维embedding")
            print(f"   模型: {result.model}")
            print(f"   Token数: {result.token_count}")
            print(f"   前5个值: {result.embedding[:5]}")

        # 测试批量embedding
        print("\n3. 测试批量embedding:")
        test_texts = [
            "第一个测试文本",
            "第二个测试文本，稍长一些",
            "第三个测试文本，用于验证批量处理功能"
        ]
        batch_results = service.get_embeddings_batch(test_texts)

        successful = sum(1 for r in batch_results if not r.error)
        failed = len(batch_results) - successful
        print(f"   批量处理结果: {successful}成功, {failed}失败")

        # 关闭服务
        service.close()
        print("\n4. 服务关闭: 成功")

    except Exception as e:
        print(f"1. 服务初始化: 失败")
        print(f"   错误: {str(e)}")
        print(f"   请确保已安装必要的依赖:")
        print(f"   pip install torch transformers sentence-transformers")

    print("\n" + "="*60)
    print("本地embedding功能测试完成")
    print("="*60)


def main():
    """主测试函数"""
    print("PDF解析器 - 本地embedding功能测试")
    print("="*60)

    test_embedding_service_structure()
    test_pdf_parser_integration()
    test_local_embedding_functionality()

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("1. embedding_service.py: 已创建（本地模型版本）")
    print("2. pdf_parser.py: 已集成本地embedding功能")
    print("3. 依赖项: 需要安装torch, transformers, sentence-transformers")
    print("4. 模型: 使用bge-base-zh-v1.5等本地模型")
    print("5. 测试PDF文件: test.pdf")
    print("\n下一步:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 创建测试文件: 将PDF文件重命名为test.pdf")
    print("3. 运行测试: python test_embedding.py")
    print("4. 运行PDF解析器: python pdf_parser.py")
    print("="*60)


if __name__ == "__main__":
    main()

