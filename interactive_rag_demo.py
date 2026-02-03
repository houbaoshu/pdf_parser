#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEA RAG系统交互式Demo
支持命令行实时问答
"""

import os
import sys
from typing import Optional
from gea_rag_agent_openai import GEARAGAgent, print_rag_answer

# ANSI颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """打印欢迎banner"""
    banner = f"""
{Colors.CYAN}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║            🤖 GEA RAG 智能问答系统 v1.0                        ║
║                                                                ║
║        基于 Chroma 向量检索 + OpenAI GPT-4o-mini              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.YELLOW}📚 知识库: 1059个GEA设备文档片段{Colors.ENDC}
{Colors.YELLOW}🔍 检索引擎: Chroma向量数据库{Colors.ENDC}
{Colors.YELLOW}🧠 AI模型: OpenAI GPT-4o-mini{Colors.ENDC}
"""
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = f"""
{Colors.BOLD}可用命令:{Colors.ENDC}

  {Colors.GREEN}/help{Colors.ENDC}      - 显示此帮助信息
  {Colors.GREEN}/examples{Colors.ENDC}  - 显示示例问题
  {Colors.GREEN}/stats{Colors.ENDC}     - 显示系统统计信息
  {Colors.GREEN}/clear{Colors.ENDC}     - 清空对话历史
  {Colors.GREEN}/config{Colors.ENDC}    - 显示当前配置
  {Colors.GREEN}/quit{Colors.ENDC}      - 退出系统 (或按 Ctrl+C)

{Colors.BOLD}配置命令:{Colors.ENDC}

  {Colors.GREEN}/topk <数字>{Colors.ENDC}        - 设置检索文档数量 (默认: 5)
  {Colors.GREEN}/temp <数字>{Colors.ENDC}        - 设置生成温度 0-1 (默认: 0.7)
  {Colors.GREEN}/tokens <数字>{Colors.ENDC}      - 设置最大生成tokens (默认: 1000)
  {Colors.GREEN}/type <类型>{Colors.ENDC}        - 限制文档类型: text/table/image/all

{Colors.BOLD}提问技巧:{Colors.ENDC}

  ✅ 具体明确: "TPS 2030的转速是多少?"
  ✅ 包含关键词: "如何更换机械密封?"
  ✅ 分步提问: "设备有哪些安全注意事项?"
  ❌ 避免模糊: "怎么样?" "好不好?"
"""
    print(help_text)


def print_examples():
    """打印示例问题"""
    examples = f"""
{Colors.BOLD}示例问题:{Colors.ENDC}

{Colors.CYAN}📊 技术参数查询:{Colors.ENDC}
  • TPS系列泵的转速是多少？
  • GEA设备的技术规格有哪些？
  • 2030型号的液体容量是多少？

{Colors.CYAN}🔧 操作维护:{Colors.ENDC}
  • 如何维护和保养GEA设备？
  • 更换机械密封需要哪些步骤？
  • 设备出现故障时如何排查？

{Colors.CYAN}⚠️ 安全注意事项:{Colors.ENDC}
  • 操作GEA设备需要注意什么安全事项？
  • 拆卸设备前需要做哪些准备？
  • 如何正确处理密封液？

{Colors.CYAN}📦 零部件信息:{Colors.ENDC}
  • O形环的材质是什么？
  • 机械密封套件包含哪些部件？
  • 如何选择正确的零部件？
"""
    print(examples)


def print_stats(agent: GEARAGAgent):
    """打印系统统计信息"""
    stats = agent.qa_agent.retriever.get_statistics()

    stats_text = f"""
{Colors.BOLD}系统统计信息:{Colors.ENDC}

  总文档数: {Colors.GREEN}{stats.get('total_chunks', 0)}{Colors.ENDC}

  文档类型分布:"""

    type_dist = stats.get('type_distribution', {})
    for doc_type, count in type_dist.items():
        stats_text += f"\n    • {doc_type}: {Colors.YELLOW}{count}{Colors.ENDC}"

    stats_text += f"""

  向量数据库: {Colors.CYAN}Chroma{Colors.ENDC}
  Embedding模型: {Colors.CYAN}BGE-base-zh-v1.5{Colors.ENDC}
  生成模型: {Colors.CYAN}{agent.model}{Colors.ENDC}
"""
    print(stats_text)


def print_config(top_k: int, temperature: float, max_tokens: int, chunk_types: Optional[list]):
    """打印当前配置"""
    config_text = f"""
{Colors.BOLD}当前配置:{Colors.ENDC}

  检索文档数 (top_k): {Colors.GREEN}{top_k}{Colors.ENDC}
  生成温度 (temperature): {Colors.GREEN}{temperature}{Colors.ENDC}
  最大tokens (max_tokens): {Colors.GREEN}{max_tokens}{Colors.ENDC}
  文档类型过滤: {Colors.GREEN}{chunk_types if chunk_types else 'all'}{Colors.ENDC}
"""
    print(config_text)


def print_answer_compact(answer):
    """紧凑格式打印答案"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}🤖 回答:{Colors.ENDC}")
    print(f"{Colors.CYAN}{answer.answer}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}📚 来源: {len(answer.sources)}个文档{Colors.ENDC}")
    for i, source in enumerate(answer.sources[:3], 1):  # 只显示前3个
        print(f"  {i}. {os.path.basename(source['source_file'])} - "
              f"第{source['page']}页 "
              f"({source['type']}, 相似度: {source['similarity']:.3f})")

    if len(answer.sources) > 3:
        print(f"  ... 还有 {len(answer.sources) - 3} 个来源")

    if answer.tokens_used:
        cost = (answer.tokens_used / 1_000_000) * 0.75  # 粗略估算成本
        print(f"\n{Colors.YELLOW}💰 Tokens: {answer.tokens_used} (~${cost:.4f}){Colors.ENDC}")
    print()


def main():
    """主函数"""
    # 打印banner
    print_banner()

    # 初始化Agent
    print(f"{Colors.YELLOW}正在初始化系统...{Colors.ENDC}")
    agent = GEARAGAgent(model="gpt-4o-mini")

    if not agent.initialize():
        print(f"{Colors.RED}❌ 初始化失败{Colors.ENDC}")
        return

    print(f"{Colors.GREEN}✅ 系统初始化成功！{Colors.ENDC}\n")
    print(f"{Colors.BOLD}输入问题开始对话，输入 /help 查看帮助{Colors.ENDC}\n")

    # 配置参数
    top_k = 5
    temperature = 0.7
    max_tokens = 1000
    chunk_types = None

    # 对话循环
    conversation_history = []
    question_count = 0

    while True:
        try:
            # 获取用户输入
            user_input = input(f"{Colors.BOLD}👤 你: {Colors.ENDC}").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.startswith('/'):
                command_parts = user_input.split()
                command = command_parts[0].lower()

                if command == '/help':
                    print_help()

                elif command == '/examples':
                    print_examples()

                elif command == '/stats':
                    print_stats(agent)

                elif command == '/clear':
                    conversation_history = []
                    question_count = 0
                    print(f"{Colors.GREEN}✅ 对话历史已清空{Colors.ENDC}\n")

                elif command == '/config':
                    print_config(top_k, temperature, max_tokens, chunk_types)

                elif command == '/quit' or command == '/exit':
                    print(f"\n{Colors.CYAN}👋 感谢使用！再见！{Colors.ENDC}\n")
                    break

                elif command == '/topk':
                    if len(command_parts) > 1:
                        try:
                            top_k = int(command_parts[1])
                            print(f"{Colors.GREEN}✅ 已设置 top_k = {top_k}{Colors.ENDC}\n")
                        except ValueError:
                            print(f"{Colors.RED}❌ 无效的数字{Colors.ENDC}\n")
                    else:
                        print(f"{Colors.RED}❌ 用法: /topk <数字>{Colors.ENDC}\n")

                elif command == '/temp':
                    if len(command_parts) > 1:
                        try:
                            temperature = float(command_parts[1])
                            if 0 <= temperature <= 1:
                                print(f"{Colors.GREEN}✅ 已设置 temperature = {temperature}{Colors.ENDC}\n")
                            else:
                                print(f"{Colors.RED}❌ 温度必须在 0-1 之间{Colors.ENDC}\n")
                        except ValueError:
                            print(f"{Colors.RED}❌ 无效的数字{Colors.ENDC}\n")
                    else:
                        print(f"{Colors.RED}❌ 用法: /temp <数字>{Colors.ENDC}\n")

                elif command == '/tokens':
                    if len(command_parts) > 1:
                        try:
                            max_tokens = int(command_parts[1])
                            print(f"{Colors.GREEN}✅ 已设置 max_tokens = {max_tokens}{Colors.ENDC}\n")
                        except ValueError:
                            print(f"{Colors.RED}❌ 无效的数字{Colors.ENDC}\n")
                    else:
                        print(f"{Colors.RED}❌ 用法: /tokens <数字>{Colors.ENDC}\n")

                elif command == '/type':
                    if len(command_parts) > 1:
                        type_arg = command_parts[1].lower()
                        if type_arg == 'all':
                            chunk_types = None
                            print(f"{Colors.GREEN}✅ 已取消类型过滤{Colors.ENDC}\n")
                        elif type_arg in ['text', 'table', 'image']:
                            chunk_types = [type_arg]
                            print(f"{Colors.GREEN}✅ 已设置类型过滤: {type_arg}{Colors.ENDC}\n")
                        else:
                            print(f"{Colors.RED}❌ 无效的类型，请使用: text/table/image/all{Colors.ENDC}\n")
                    else:
                        print(f"{Colors.RED}❌ 用法: /type <text|table|image|all>{Colors.ENDC}\n")

                else:
                    print(f"{Colors.RED}❌ 未知命令: {command}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}输入 /help 查看可用命令{Colors.ENDC}\n")

                continue

            # 处理问题
            question_count += 1
            print(f"\n{Colors.YELLOW}⏳ 正在思考...{Colors.ENDC}")

            # 执行查询
            answer = agent.query(
                question=user_input,
                top_k=top_k,
                chunk_types=chunk_types,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 显示答案
            print_answer_compact(answer)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}👋 检测到中断，正在退出...{Colors.ENDC}\n")
            break

        except Exception as e:
            print(f"\n{Colors.RED}❌ 发生错误: {str(e)}{Colors.ENDC}\n")
            continue

    # 显示统计
    if question_count > 0:
        print(f"{Colors.BOLD}📊 会话统计:{Colors.ENDC}")
        print(f"  总提问数: {Colors.GREEN}{question_count}{Colors.ENDC}")
        print()


if __name__ == "__main__":
    main()
