import os
import socket
import streamlit as st
import requests

def get_api_url():
    # 尝试解析 'backend'，如果成功说明在 Docker 网络内
    try:
        socket.gethostbyname("backend")
        return "http://backend:8000"
    except socket.gaierror:
        return "http://localhost:8000"

API_URL = get_api_url()

st.set_page_config(page_title="🏢 企业报销助手", page_icon="🤖")
st.title("🤖 企业智能报销助手")
st.caption("基于 RAG 技术，由本地大模型驱动")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请问关于报销有什么问题？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在请求后端..."):
            answer = "抱歉，暂时无法获取回答。"
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt, "conversation_id": "default"},
                    timeout=120
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    with st.expander("查看参考来源"):
                        for i, src in enumerate(data.get("sources", [])):
                            st.write(f"**片段 {i+1}:** {src}")
                else:
                    st.error(f"API 请求失败，状态码: {resp.status_code}")
            except Exception as e:
                st.error(f"连接后端失败: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer})