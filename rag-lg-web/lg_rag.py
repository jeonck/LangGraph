from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from pathlib import Path
import os

# FAISS 저장 경로 설정
PERSIST_DIRECTORY = Path(__file__).parent.parent / "faiss_index"

def initialize_vectorstore(api_key: str):
    """벡터 스토어를 초기화하거나 로드합니다."""
    try:
        print("기존 FAISS 인덱스 로드 시도...")
        vectorstore = FAISS.load_local(
            folder_path=str(PERSIST_DIRECTORY),
            embeddings=OpenAIEmbeddings(openai_api_key=api_key),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"기존 인덱스 로드 실패, 새로 생성합니다: {e}")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        documents = [
            Document(page_content="KBO 리그는 대한민국의 프로야구 리그입니다."),
            Document(page_content="MLB는 메이저리그 베이스볼의 약자입니다."),
            Document(page_content="NPB는 일본의 프로야구 리그입니다."),
        ]
        
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=api_key))
        vectorstore.save_local(str(PERSIST_DIRECTORY))
    
    return vectorstore

def ask_question(question: str, api_key: str) -> str:
    """질문에 대한 답변을 생성합니다."""
    try:
        # 매번 vectorstore 초기화
        vectorstore = initialize_vectorstore(api_key)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain({"query": question})
        return result["result"]
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"
