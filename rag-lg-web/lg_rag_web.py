import streamlit as st
from lg_rag import ask_question

st.title("야구 관련 질문하기")

# API 키 입력 받기
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")

# API 키가 입력된 경우에만 질문 입력 허용
if api_key:
    # 사용자 입력 받기
    user_question = st.text_input("질문을 입력하세요:")

    # 질문이 입력되면 답변 생성
    if user_question:
        try:
            answer = ask_question(user_question, api_key)  # API 키 전달
            st.write("답변:", answer)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
else:
    st.warning("계속하려면 OpenAI API 키를 입력해주세요.") 