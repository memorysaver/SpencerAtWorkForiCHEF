import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
loader = NotionDBLoader(NOTION_API_KEY, NOTION_DATABASE_ID)


llm = OpenAI(model_name="gpt-3.5-turbo",max_tokens=1024)
embeddings = OpenAIEmbeddings()
persist_directory = 'chroma.db'

def reset_vectordb():
    docs = loader.load()
    # st.write(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 0
    )

    split_docs = text_splitter.split_documents(docs)
    st.write(split_docs)


    # fix for notion db
    texts = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]
    metadatas = [{**d, "tags": ",".join(d["tags"])} for d in metadatas]
    st.write(texts)
    st.write(metadatas)

    vectordb = Chroma.from_texts(texts, embeddings, metadatas=metadatas,persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

st.sidebar.button('重置資料庫', on_click=reset_vectordb)

def clear_talks():
    st.session_state.past = []
    st.session_state.generated = []
st.sidebar.button('清除對話', on_click=clear_talks)



# load from db
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


template = """
    我希望你可以扮演我，代替我處理本週日常的工作，我會提供我的工作的範疇以及處理的原則，並且我會盡可能的提供工作所需要的背景知識。
    如果被要求工作內容提供的背景不足，可以跟對方直接詢問細節，但是跟我設定的工作範圍之外的問題都請委婉的拒絕回答。
    請勿隨意回答你不知道並且沒有事實根據的問題。

    ＃context
    -------------
    {context}
    -----------

    # chat history
    {chat_history}

    現在開始扮演我，來回答接下來的各種問題，我們直接開始。
    Question: {human_input}
    Answer:
"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

def query(query_str):
    # https://python.langchain.com/en/latest/modules/memory/examples/adding_memory_chain_multiple_inputs.html
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
    query_docs = vectordb.similarity_search(query_str)
    result =  chain({"input_documents": query_docs, "human_input": query_str}, return_only_outputs=False)
    st.sidebar.write(query_docs)
    st.sidebar.write(result)
    st.sidebar.write(chain.memory.buffer)
    return result['output_text']

def submit():
    output = query(st.session_state.input)
    st.session_state.past.append(st.session_state.input)
    st.session_state.generated.append(f"{output}")
    st.session_state.current_input = st.session_state.input
    st.session_state.input = ''

st.text_input('You: ', key='input', on_change=submit)

if 'current_input' not in st.session_state:
    st.session_state.current_input = ''

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


