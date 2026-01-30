# Python 3.10 虚拟环境使用说明

## 虚拟环境信息

- **Python版本**: 3.10.19
- **虚拟环境路径**: `venv/`
- **创建时间**: 2026-01-30
- **基础Python**: Homebrew安装的Python 3.10

## 使用方法

### 1. 激活虚拟环境

在项目根目录下运行：

```bash
# 在Unix/Linux/macOS上
source venv/bin/activate

# 在Windows上（如果使用Git Bash）
source venv/Scripts/activate

# 在Windows上（如果使用Command Prompt）
venv\Scripts\activate
```

激活后，命令行提示符会显示 `(venv)` 前缀。

### 2. 验证虚拟环境

激活后，可以运行以下命令验证：

```bash
python --version
# 应该显示: Python 3.10.19

which python
# 应该显示虚拟环境中的Python路径

pip --version
# 应该显示虚拟环境中的pip版本
```

或者运行测试脚本：

```bash
python test_venv.py
```

### 3. 安装Python包

在虚拟环境中，使用pip安装包：

```bash
pip install <package-name>
```

例如：

```bash
pip install pandas numpy matplotlib
```

### 4. 生成requirements.txt

要保存当前虚拟环境中安装的所有包：

```bash
pip freeze > requirements.txt
```

### 5. 从requirements.txt安装

要在新的环境中安装所有依赖：

```bash
pip install -r requirements.txt
```

### 6. 退出虚拟环境

```bash
deactivate
```

## 虚拟环境管理

### 查看已安装的包

```bash
pip list
```

### 升级pip

```bash
pip install --upgrade pip
```

### 删除虚拟环境

如果需要重新创建虚拟环境：

```bash
# 先退出虚拟环境
deactivate

# 删除虚拟环境目录
rm -rf venv/

# 重新创建虚拟环境
python3.10 -m venv venv
```

## 注意事项

1. **不要将虚拟环境目录提交到版本控制**：建议将 `venv/` 添加到 `.gitignore` 文件中。
2. **跨平台兼容性**：虚拟环境通常与创建它的操作系统和Python版本绑定。
3. **Python版本**：此虚拟环境使用Python 3.10.19创建，确保项目代码与此版本兼容。

## 故障排除

### 问题：无法激活虚拟环境
- 确保在项目根目录下
- 确保 `venv/` 目录存在
- 检查文件权限：`chmod +x venv/bin/activate`

### 问题：Python版本不正确
- 检查是否已激活虚拟环境（命令行应有 `(venv)` 前缀）
- 运行 `which python` 确认使用的是虚拟环境中的Python

### 问题：pip命令不可用
- 尝试重新创建虚拟环境：`python3.10 -m venv venv`
- 检查Python安装是否完整

