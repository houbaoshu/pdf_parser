# GEA RAG Web应用使用指南

## 🎯 概述

这是一个基于Streamlit构建的Web界面，为GEA RAG问答系统提供友好的交互体验。

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
# 激活虚拟环境
source .venv_debian/bin/activate

# 安装依赖（如果还没装）
pip install streamlit streamlit-chat
```

或者更新整个环境：

```bash
pip install -r requirements.txt
```

### 步骤2: 设置OpenAI API密钥

```bash
export OPENAI_API_KEY="your-api-key-here"
```

或者在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
echo 'export OPENAI_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 步骤3: 启动Web应用

```bash
streamlit run streamlit_app.py
```

应用会自动在浏览器中打开，默认地址是 `http://localhost:8501`

## 📱 界面功能

### 主界面

1. **聊天窗口**
   - 显示历史对话记录
   - 支持多轮对话
   - 实时显示问答结果

2. **输入框**
   - 底部输入框输入问题
   - 按Enter或点击发送按钮提交

3. **来源文档**
   - 每个回答下方可展开查看来源
   - 显示文件名、页码、类型、相似度
   - 内容预览

4. **Tokens使用**
   - 显示每次查询的tokens消耗
   - 估算成本（基于GPT-4o-mini价格）

### 侧边栏配置

#### 检索设置

- **检索文档数量 (top_k)**: 1-10
  - 数量越多，上下文越丰富，但速度越慢
  - 推荐: 3-5

- **文档类型过滤**:
  - 全部: 搜索所有类型
  - 文本: 只搜索文本chunks
  - 表格: 只搜索表格数据
  - 图像: 只搜索图像描述

#### 生成设置

- **生成温度 (temperature)**: 0.0-1.0
  - 0.0-0.3: 更精确，适合技术参数查询
  - 0.4-0.7: 平衡精确性和流畅性（推荐）
  - 0.8-1.0: 更有创意，但可能不够精确

- **最大生成Tokens**: 100-2000
  - 控制回答长度
  - 推荐: 1000

#### 系统统计

- 显示总文档数
- 显示各类型文档分布

#### 操作按钮

- **清空对话**: 重置会话，开始新对话

## 💡 使用技巧

### 1. 提问技巧

✅ **好的问题**:
- "TPS 2030型号的转速是多少？"
- "如何更换机械密封？"
- "设备维护需要注意哪些安全事项？"

❌ **不好的问题**:
- "设备怎么样？"（太模糊）
- "所有参数"（太宽泛）
- "价格多少？"（文档中没有）

### 2. 配置优化

**快速查询**（精确问题）:
- top_k: 3
- temperature: 0.3
- max_tokens: 500

**详细查询**（综合问题）:
- top_k: 7
- temperature: 0.7
- max_tokens: 1500

**表格数据查询**:
- 文档类型: 表格
- top_k: 5
- temperature: 0.3

### 3. 多轮对话

系统会自动保持对话历史，你可以：
- 提问后继续追问细节
- 基于之前的回答提问
- 使用"它"、"这个"等代词指代前文

示例对话：
```
用户: GEA设备有哪些型号？
助手: [列出型号...]

用户: TPS 2030的详细参数是什么？
助手: [详细参数...]

用户: 它的维护周期是多久？
助手: [维护信息...]
```

## 🎨 界面说明

### 相似度颜色标识

- 🟢 **绿色** (≥0.6): 高相关性，答案可信度高
- 🟡 **黄色** (0.4-0.6): 中等相关性，需要结合上下文
- 🔴 **红色** (<0.4): 低相关性，答案可能不准确

### 来源文档框

每个来源文档显示：
- 📄 文件名
- 📑 页码
- 📝 类型（text/table/image）
- 🎯 相似度分数
- 📖 内容预览（前100字）

## ⚙️ 高级配置

### 自定义端口

```bash
streamlit run streamlit_app.py --server.port 8080
```

### 允许外部访问

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### 配置文件

创建 `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

## 🐛 故障排除

### 问题1: 应用无法启动

**可能原因**: 缺少依赖

**解决方案**:
```bash
pip install streamlit streamlit-chat
```

### 问题2: OpenAI API错误

**可能原因**: API密钥未设置或无效

**解决方案**:
```bash
# 检查API密钥
echo $OPENAI_API_KEY

# 设置API密钥
export OPENAI_API_KEY="your-api-key"
```

### 问题3: 端口已被占用

**错误信息**: `Address already in use`

**解决方案**:
```bash
# 使用不同端口
streamlit run streamlit_app.py --server.port 8502
```

### 问题4: 初始化失败

**可能原因**: Chroma数据库不存在

**解决方案**:
```bash
# 运行迁移脚本
python migrate_to_chroma.py
```

### 问题5: 回答太慢

**优化方案**:
1. 减少 top_k (3-5)
2. 减少 max_tokens (500-800)
3. 确保网络连接良好

## 📊 性能指标

| 操作 | 平均时间 |
|------|---------|
| 页面加载 | ~2秒 |
| Agent初始化 | ~1秒 |
| 检索文档 | ~0.2秒 |
| 生成答案 | ~2-5秒 |
| 总响应时间 | ~3-7秒 |

## 🌐 部署选项

### 本地部署

适合个人使用或团队内部使用：
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud部署

1. 将代码推送到GitHub
2. 访问 https://share.streamlit.io/
3. 连接GitHub仓库
4. 在Secrets中设置 `OPENAI_API_KEY`
5. 部署

### Docker部署

创建 `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

构建和运行：
```bash
docker build -t gea-rag-app .
docker run -p 8501:8501 -e OPENAI_API_KEY="your-key" gea-rag-app
```

## 🔐 安全建议

1. **API密钥安全**:
   - 不要在代码中硬编码API密钥
   - 使用环境变量或secrets管理
   - 定期轮换API密钥

2. **访问控制**:
   - 在生产环境中添加身份验证
   - 限制访问IP范围
   - 使用HTTPS

3. **数据隐私**:
   - 不要在公共环境中讨论敏感信息
   - 定期清理对话历史
   - 注意OpenAI数据政策

## 📈 使用场景

### 场景1: 技术支持

技术人员快速查询设备参数和维护信息

**推荐配置**:
- top_k: 3
- temperature: 0.3
- 文档类型: 文本+表格

### 场景2: 培训学习

新员工学习GEA设备知识

**推荐配置**:
- top_k: 5
- temperature: 0.7
- 文档类型: 全部

### 场景3: 数据分析

查询和对比不同型号的技术参数

**推荐配置**:
- top_k: 7
- temperature: 0.3
- 文档类型: 表格

## 🎓 下一步

完成Web应用后，可以继续优化：

1. ✅ **添加用户认证** - 使用Streamlit Auth
2. ✅ **对话导出** - 导出问答历史为PDF/Markdown
3. ✅ **数据分析** - 统计常见问题和使用情况
4. ✅ **Re-ranking** - 集成BGE-reranker提升准确率
5. ✅ **反馈系统** - 收集用户反馈改进答案质量

## 📞 需要帮助？

- 查看示例: 启动应用后点击侧边栏"提问技巧"
- Streamlit文档: https://docs.streamlit.io/
- 项目文档: 查看其他*.md文件

---

**开始使用吧！** 🚀

```bash
streamlit run streamlit_app.py
```
