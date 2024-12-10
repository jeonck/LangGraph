# lg_rag.py
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import os
import logging
from datetime import datetime

load_dotenv()

# 환경 변수 설정
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# [지식 업데이트 시 수정 필요] - 크롤링할 URL 목록
urls = [
    "https://namu.wiki/w/%EC%95%BC%EA%B5%AC/%EA%B2%BD%EA%B8%B0%20%EB%B0%A9%EC%8B%9D",
]

# WebBaseLoader 생성 시 header_template 사용
headers = {
    "User-Agent": os.environ["USER_AGENT"]  # 환경 변수에서 가져온 값 사용
}

docs = [WebBaseLoader(url, header_template=headers).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0, model_name="gpt-4o-mini"
)
doc_splits = text_splitter.split_documents(docs_list)

# [지식 업데이트 시 수정 필요] - 벡터 스토어 설정
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="baseball-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma",
)

# [지식 업데이트 시 수정 필요] - 검색기 설정
retriever = Chroma(
    collection_name="baseball-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma",
).as_retriever()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str  # 사용자 질문 저장
    context: str  # 검색된 문서 저장
    response: str  # 최종 응답 저장

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_trace.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def should_retrieve(state: AgentState) -> dict:
    logger.info("=== 검색 단계 시작 ===")
    query = state["query"]
    logger.info(f"검색 쿼리: {query}")
    
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    logger.info(f"검색된 문서 수: {len(docs)}")
    logger.info("=== 검색된 문서 목록 ===")
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n문서 {i}:")
        logger.info(f"내용: {doc.page_content}")
        logger.info(f"메타데이터: {doc.metadata}")
        logger.info("-" * 50)
    logger.info("=== 검색 단계 완료 ===")
    
    return {"next": "grade_documents", "context": context}

def grade_documents(state: AgentState) -> dict:
    logger.info("=== 문서 평가 단계 시작 ===")
    context = state["context"]
    if not context:
        logger.warning("컨텍스트가 비어있음 - 쿼리 재작성 필요")
        return {"next": "rewrite_query"}
    logger.info("문서 평가 완료 - 답변 생성 단계로 진행")
    return {"next": "generate_answer"}

def rewrite_query(state: AgentState) -> dict:
    # 쿼리 재작성
    return None

# [지식 업데이트 시 수정 필요] - 프롬프트 템플릿
def generate_answer(state: AgentState) -> dict:
    logger.info("=== 답변 생성 단계 시작 ===")
    llm = ChatOpenAI(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 정보를 바탕으로 사용자의 질문에 답변해주세요:\n\n{context}"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    logger.info(f"입력 쿼리: {state['query']}")
    logger.info(f"사용된 컨텍스트 길이: {len(state['context'])} 문자")
    
    response = chain.invoke({
        "context": state["context"],
        "query": state["query"]
    })
    
    logger.info(f"생성된 답변: {response.content}")
    logger.info("=== 답변 생성 단계 완료 ===")
    
    return {"response": response.content}

# StateGraph 설정
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("should_retrieve", should_retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate_answer", generate_answer)

# 시작점 설정
graph.set_entry_point("should_retrieve")

# 엣지 추가
graph.add_edge("should_retrieve", "grade_documents")
graph.add_edge("grade_documents", "generate_answer")
graph.add_edge("grade_documents", "rewrite_query")

# 그래프 컴파일
chain = graph.compile()

# 질문-답변 실행 함수
def ask_question(question: str):
    logger.info("\n=== 새로운 질문 처리 시작 ===")
    logger.info(f"입력된 질문: {question}")
    
    start_time = datetime.now()
    
    initial_state = AgentState(
        messages=[],
        query=question,
        context="",
        response=""
    )
    
    result = chain.invoke(initial_state)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(f"처리 시간: {processing_time:.2f}초")
    logger.info("=== 질문 처리 완료 ===\n")
    
    return result["response"]

# 사용 예시
# question = "야구 경기에서 투수가 던질 수 있는 최대 투구수는 몇 개인가요?"
# answer = ask_question(question)
# print(f"질문: {question}")
# print(f"답변: {answer}")
