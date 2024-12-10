# app.py
import streamlit as st
from lg_rag import ask_question

# [지식 업데이트 시 수정 필요] - 페이지 설정
st.set_page_config(
    page_title="야구 지식 챗봇",  # 제목 변경 가능
    page_icon="⚾"  # 아이콘 변경 가능
)
st.title("⚾ 야구 지식 챗봇")  # 제목 변경 가능


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            response = ask_question(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 
