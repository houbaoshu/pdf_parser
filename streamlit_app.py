#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEA RAG系统 - Streamlit Web界面
提供友好的Web交互界面进行问答
"""

import os
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from gea_rag_agent_openai import GEARAGAgent, RAGAnswer


# 页面配置
st.set_page_config(
    page_title="GEA RAG 智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .source-box {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        border-left: 5px solid #1f77b4;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .source-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .similarity-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
    .similarity-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
    .similarity-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
    .stat-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1e9f0 100%);
        padding: 1.2rem;
        border-radius: 1rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(31, 119, 180, 0.2);
    }
    .stChatMessage {
        border-radius: 1.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agent():
    """初始化RAG Agent（缓存）"""
    agent = GEARAGAgent(model="gpt-4o-mini")
    if agent.initialize():
        return agent
    return None


def get_similarity_class(similarity: float) -> str:
    """根据相似度返回CSS类名"""
    if similarity >= 0.6:
        return "similarity-high"
    elif similarity >= 0.4:
        return "similarity-medium"
    else:
        return "similarity-low"


def format_source(source: Dict[str, Any], index: int) -> str:
    """格式化来源文档"""
    sim_class = get_similarity_class(source['similarity'])

    html = f"""
    <div class="source-box">
        <strong>📄 来源 {index}</strong><br>
        <strong>文件:</strong> {os.path.basename(source['source_file'])}<br>
        <strong>页码:</strong> {source['page']} |
        <strong>类型:</strong> {source['type']} |
        <strong>相似度:</strong> <span class="{sim_class}">{source['similarity']:.3f}</span><br>
        <strong>内容预览:</strong> {source['content_preview']}
    </div>
    """
    return html


def display_message(role: str, content: str, answer: RAGAnswer = None):
    """显示消息"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

            # 如果有完整答案对象，显示额外信息
            if answer and answer.has_answer:
                # 显示来源
                with st.expander(f"📚 查看来源文档 ({len(answer.sources)}个)", expanded=False):
                    for i, source in enumerate(answer.sources, 1):
                        st.markdown(format_source(source, i), unsafe_allow_html=True)

                # 显示tokens使用
                if answer.tokens_used:
                    cost = (answer.tokens_used / 1_000_000) * 0.75
                    st.caption(f"💰 Tokens: {answer.tokens_used} (~${cost:.4f})")


def main():
    """主函数"""
    # 标题
    st.markdown('<div class="main-header">🤖 GEA RAG 智能问答系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于 Chroma 向量检索 + OpenAI GPT-4o-mini</div>', unsafe_allow_html=True)

    # 初始化会话状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'agent' not in st.session_state:
        with st.spinner("正在初始化RAG Agent..."):
            agent = initialize_agent()
            if agent is None:
                st.error("❌ 初始化失败，请检查配置和环境变量（OPENAI_API_KEY）")
                st.stop()
            st.session_state.agent = agent
            st.success("✅ RAG Agent初始化成功！")

    agent = st.session_state.agent

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置选项")
        
        # API设置
        with st.expander("🔑 API 设置", expanded=False):
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                # 如果Agent已经初始化但Key变了，重新初始化
                if 'agent' in st.session_state and st.session_state.agent.client.api_key != api_key:
                    st.session_state.agent = None
                    st.rerun()

        # 检索配置
        st.subheader("检索设置")
        top_k = st.slider("检索文档数量 (top_k)", min_value=1, max_value=10, value=5, step=1)

        chunk_type = st.selectbox(
            "文档类型过滤",
            options=["全部", "文本", "表格", "图像"],
            index=0
        )
        chunk_types = None
        if chunk_type == "文本":
            chunk_types = ["text"]
        elif chunk_type == "表格":
            chunk_types = ["table"]
        elif chunk_type == "图像":
            chunk_types = ["image"]

        # 生成配置
        st.subheader("生成设置")
        temperature = st.slider(
            "生成温度 (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="较低的值更精确，较高的值更有创意"
        )

        max_tokens = st.slider(
            "最大生成Tokens",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )

        st.divider()

        # 系统统计
        st.subheader("📊 系统统计")
        if hasattr(agent, 'qa_agent'):
            stats = agent.qa_agent.retriever.get_statistics()

            st.markdown(f"""
            <div class="stat-box">
            <strong>总文档数:</strong> {stats.get('total_chunks', 0)}<br>
            <strong>文本:</strong> {stats.get('type_distribution', {}).get('text', 0)}<br>
            <strong>表格:</strong> {stats.get('type_distribution', {}).get('table', 0)}<br>
            <strong>图像:</strong> {stats.get('type_distribution', {}).get('image', 0)}
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 操作按钮
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # 帮助信息
        with st.expander("💡 提问技巧"):
            st.markdown("""
            **好的问题示例:**
            - TPS 2030的转速是多少？
            - 如何更换机械密封？
            - 设备维护需要注意哪些安全事项？

            **避免:**
            - 太模糊的问题（"怎么样？"）
            - 文档中没有的内容（"价格多少？"）
            """)

    # 显示历史消息
    for msg in st.session_state.messages:
        display_message(
            msg["role"],
            msg["content"],
            msg.get("answer")  # 完整答案对象（如果有）
        )

    # 用户输入
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        display_message("user", prompt)

        # 生成回答
        with st.spinner("🤔 正在思考..."):
            try:
                answer = agent.query(
                    question=prompt,
                    top_k=top_k,
                    chunk_types=chunk_types,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # 添加助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer.answer,
                    "answer": answer  # 保存完整答案对象
                })

                # 显示回答
                display_message("assistant", answer.answer, answer)

            except Exception as e:
                error_msg = f"❌ 发生错误: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # 页脚
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔍 向量检索: Chroma")
    with col2:
        st.caption("🧠 Embedding: BGE-base-zh-v1.5")
    with col3:
        st.caption("💬 LLM: GPT-4o-mini")


if __name__ == "__main__":
    main()
