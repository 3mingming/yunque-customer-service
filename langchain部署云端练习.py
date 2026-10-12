from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool
from langchain.agents import create_agent
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import DashScopeRerank
import dashscope
from dashscope import TextReRank



load_dotenv("C:/Users/30683/PyCharmMiscProject/my_models/env",override=True)
api_key=os.getenv("embedding_api_key")
base_url=os.getenv("embedding_base_url")
embedding_model=init_embeddings(model='qwen3.7-text-embedding-flash',
                                api_key=api_key,
                                base_url=base_url,
                                provider='openai',
                                check_embedding_ctx_length=False)
model_api_key=os.getenv('api_key')
model_base_url=os.getenv('base_url')
model=ChatOpenAI(model='qwen-plus',
                 api_key=model_api_key,
                 base_url=model_base_url)
loader=DirectoryLoader("data/",glob='**/*.md',
                       loader_cls=TextLoader,
                       loader_kwargs={"autodetect_encoding":True})
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
splitts=text_splitter.split_documents(docs)
store=Chroma.from_documents(documents=splitts,embedding=embedding_model)

base_retriever=store.as_retriever(search_kwargs={"k":20})
conn=sqlite3.connect('checkpoint.db',check_same_thread=False)
memory=SqliteSaver(conn)
def format_docs(docs_with_source):
    formatted=[]
    for doc,source in docs_with_source:
        source_name=os.path.basename(doc.metadata.get('source','未知文件'))
        formatted.append(f"【来源：{source_name}【分数：{source:.4f}】\n{doc.page_content}")
        return "\n\n".join(formatted)

@tool
def calculator(expression:str):
    """
    用于计算数学表达式，输入必须是合法的表达式，例如：‘199*3+20’
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{str(e)}"
@tool
def get_current_time():
    """获取当前的系统时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@tool
def search_doc(query:str):
    """当用户询问任何关于云雀协作产品、功能、价格、退款、安装、平台支持等问题时，必须使用此工具检索文档。包括但不限于：介绍产品、产品是什么、有哪些功能、支持什么平台、多少钱、怎么退款、怎么安装"""
    condidates=store.similarity_search(query,k=20)

    if not condidates:
        return "没有检索到相关文档"
    resp=TextReRank.call(model='get-rerank-v2',query=query,
                         documents=[d.page_content for d in condidates],
                         top_n=3,
                         return_documents=False,
                         api_key=os.getenv('api_key'))
    if resp.status_code !=200 or not resp.output:
        top_docs=condidates[:3]
    else:
        top_docs=[condidates[r['index']] for r in resp.output.results]
    formatted=[]
    for doc in top_docs :
        source_name=os.path.basename(doc.metadata.get('source',"未知文件"))
        formatted.append(f"【来源：{source_name}】\n{doc.page_content}")
    return "\n\n".join(formatted)




tools=[calculator,get_current_time,search_doc]

agent = create_agent(model=model,tools=tools,

                     system_prompt='你是一个云雀协作的客服。当用户询问关于云雀协作的任何问题时，必须先调用 search_doc 工具检索文档。然后阅读工具返回的内容，用你自己的话提炼总结，直接回答用户的问题。不要照抄文档原文，不要罗列文档里的所有细节，控制在100字以内。如果工具返回"没有检索到相关的文档"，才回复"不知道"。',
checkpointer=memory)
import streamlit as st
import uuid
if "session_id" not in st.session_state:
    st.session_state.session_id=str(uuid.uuid4())
st.title("云雀协作客服")
if 'messages' not in st.session_state:
    st.session_state.messages=[]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if question:=st.chat_input("请输入问题"):
    st.session_state.messages.append({'role':'user','content':question})
    st.chat_message('user').write(question)
    stream=agent.stream({"messages":[{'role':'user','content':question}]},
                          config={'configurable':{'thread_id':st.session_state.session_id}},
                          stream_mode='messages')
    from langchain_core.messages import AIMessageChunk

    with st.chat_message('assistant'):
        def generate():
            for chunk, metadata in stream:
                # 只保留 LLM 生成的 AIMessageChunk
                if not isinstance(chunk, AIMessageChunk):
                    continue
                if not chunk.content:
                    continue
                # 再排除工具节点
                if metadata.get("langgraph_node") == "tools":
                    continue
                yield chunk.content


        answer = st.write_stream(generate())

    st.session_state.messages.append({'role': 'assistant', 'content': answer})




