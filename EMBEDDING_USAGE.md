# PDF解析器 - 本地embedding功能使用指南

## 概述

本PDF解析器集成了本地embedding功能，使用bge-base-zh-v1.5等本地模型为PDF文档的chunks生成向量表示，无需依赖云服务。

## 功能特性

1. **PDF解析**: 提取文本、表格和图像内容
2. **Chunk合并**: 将blocks合并为语义chunks
3. **本地embedding**: 使用本地模型生成向量表示
4. **批量处理**: 支持批量处理多个PDF文件
5. **结果导出**: 将结果保存为JSON格式

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- `pdfplumber`: PDF解析
- `torch`: PyTorch深度学习框架
- `transformers`: Hugging Face Transformers库
- `sentence-transformers`: Sentence Transformers库

## 支持的本地模型

1. **BGE-BASE-ZH-V1.5** (默认): BAAI/bge-base-zh-v1.5
2. **BGE-LARGE-ZH-V1.5**: BAAI/bge-large-zh-v1.5

## 使用方法

### 1. 基本使用

运行主程序处理PDF文件:

```bash
python pdf_parser.py
```

### 2. 测试embedding功能

```bash
python test_embedding.py
```

### 3. 单独处理单个PDF文件

```python
from pdf_parser import process_pdf_with_embeddings

# 处理单个PDF文件
result = process_pdf_with_embeddings(
    "test.pdf",
    max_chars_per_chunk=800,
    embedding_config={
        "local_model_name": "BAAI/bge-base-zh-v1.5",
        "device": "cpu",  # 或 "cuda" 如果有GPU
        "batch_size": 100
    }
)

# 保存结果
import json
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
```

### 4. 使用embedding服务

```python
from embedding_service import create_embedding_service, EmbeddingConfig

# 创建配置
config = EmbeddingConfig(
    local_model_name="BAAI/bge-base-zh-v1.5",
    device="cpu",
    batch_size=100
)

# 创建服务
service = create_embedding_service(config)

# 获取单个文本的embedding
result = service.get_embedding("这是一个测试文本")
if not result.error:
    print(f"维度: {result.dimensions}")
    print(f"模型: {result.model}")
    print(f"前5个值: {result.embedding[:5]}")

# 批量获取embedding
texts = ["文本1", "文本2", "文本3"]
results = service.get_embeddings_batch(texts)

# 关闭服务
service.close()
```

## 输出格式

### 1. 原始blocks结果
- `test_pdf_results.json`: 单个文件原始结果

### 2. Chunks结果
- `test_pdf_chunks_results.json`: 单个文件chunks结果

### 3. Embedding结果
- `test_pdf_embeddings_results.json`: 单个文件embedding结果

## Chunk数据结构

每个chunk包含以下字段:

```json
{
  "type": "text|table|image",
  "page": 1,
  "content": "chunk内容",
  "char_count": 123,
  "embedding": {
    "vector": [0.1, 0.2, ...],
    "model": "BAAI/bge-base-zh-v1.5",
    "dimensions": 768,
    "token_count": 10,
    "has_error": false,
    "error": null
  }
}
```

## 配置选项

### EmbeddingConfig参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | EmbeddingModel | BGE_BASE_ZH_V1_5 | embedding模型枚举 |
| local_model_name | str | "BAAI/bge-base-zh-v1.5" | 本地模型名称 |
| device | str | "cpu" | 设备：cpu或cuda |
| normalize_embeddings | bool | True | 是否归一化embedding |
| timeout | int | 30 | 请求超时时间（秒） |
| max_retries | int | 3 | 最大重试次数 |
| retry_delay | float | 1.0 | 重试延迟（秒） |
| batch_size | int | 100 | 批量处理大小 |

### 环境变量

| 变量名 | 说明 |
|--------|------|
| EMBEDDING_LOCAL_MODEL | 本地模型名称 |
| EMBEDDING_DEVICE | 设备：cpu或cuda |
| EMBEDDING_TIMEOUT | 请求超时时间 |
| EMBEDDING_BATCH_SIZE | 批量处理大小 |

## 错误处理

1. **模型加载错误**: 检查模型名称是否正确，网络连接是否正常
2. **内存不足**: 减少batch_size或使用CPU模式
3. **PDF文件错误**: 检查PDF文件是否损坏
4. **依赖缺失**: 确保已安装所有必要的依赖

## 性能优化建议

1. **使用GPU**: 如果有GPU，设置`device="cuda"`加速计算
2. **调整chunk大小**: 根据内容调整`max_chars_per_chunk`
3. **批量处理**: 使用批量处理提高效率
4. **模型选择**: 根据需求选择合适的模型
   - `bge-base-zh-v1.5`: 平衡性能和精度
   - `bge-large-zh-v1.5`: 更高精度，更大模型

## 模型特点

### BGE-BASE-ZH-V1.5
- 维度: 768
- 语言: 中文优化
- 大小: 约400MB
- 特点: 平衡性能和精度，适合大多数应用

### BGE-LARGE-ZH-V1.5
- 维度: 1024
- 语言: 中文优化
- 大小: 约1.3GB
- 特点: 更高精度，适合对质量要求高的应用

## 示例输出

```
PDF解析器 - 本地embedding版本

