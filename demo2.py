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
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
loader = NotionDBLoader(NOTION_API_KEY, NOTION_DATABASE_ID)


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
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", input_key="possible_answers")
st.sidebar.button('清除對話', on_click=clear_talks)



# load from long term and short term memory
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", input_key="possible_answers")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def generate_conversational_assumtion_chain(memory):
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory)
    pass

def generate_assumption_chain():
    template = """以下是問題的陳述:
        {question}\n

        你現在是何明政，請用以下的資訊先用簡明的方式改寫一下問題，然後依據改寫後的問題陳述，透過以下資訊列出三個問題陳述背後可能的假設，以及為什麼會有這個假設的原因，並且盡可能的列出你的假設的依據，用條列的方式呈現。
        ＃context
        -------------
        {context}
        -----------

        # chat history
        {chat_history}

        用以下格式呈現:
        改寫的問題: <改寫的問題>

        假設:
        假設的依據:
        假設的原因：
        ...
    """
    prompt = PromptTemplate(
        input_variables=["question", "context", "chat_history"], 
        template=template
    )
    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024, temperature=0.4)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def generate_answer_chain():
    template = """以下是問題的陳述以及背後的假設:
        {question}\n

        你現在是何明政，請基於以下資訊以及以上的假設，提出三個可能的回答，並且基於答案的依據以及原因給出評分，請基於事實回答。
        ＃context
        -------------
        {context}
        \n\n

        用以下格式呈現:
        目前的問題: <目前的問題>

        答案:
        答案的依據:
        答案的原因：
        答案的評分:
        ...
    """
    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=template
    )
    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024, temperature=0.2)
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
    chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectordb.as_retriever())
    return chain

def generate_final_chain():
    template = """以下是問題的陳述以及背後的假設:
        {possible_answers}\n

        你現在是何明政，可以綜合以上兩個以上最高分的答案，以及以下所提供的資訊以及過往的對話，來思考適合的答案。
        ＃context
        -------------
        {context}
        -----------

        # chat history
        {chat_history}

        先摘要要整合的答案提供給我，在總結做出完整的回答，請基於事實回答。
    """
    prompt = PromptTemplate(
        input_variables=["possible_answers", "context", "chat_history"], 
        template=template
    )
    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024, temperature=0.2)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def rephrase_like_a_human_chain():
    template = """

        # Instruction
        你現在扮演的是何明政，請基於以下 context 以及 chat history ，重新改寫整理好的答案，不要改變原本的意思，保留所有的資訊，並且以他為第一人稱的方式用他的語氣來回答。

        # 整理好的答案
        ------------------------------------
        {final_answers}\n\n
        ------------------------------------

        ＃context
        ------------------------------------
        {context}
        ------------------------------------

        # chat history
        ------------------------------------
        {chat_history}
        ------------------------------------

        直接回應改寫後的答案:
    """
    prompt = PromptTemplate(
        input_variables=["final_answers", "context", "chat_history"], 
        template=template
    )
    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024, temperature=0.4)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def query(query_str):
    # https://python.langchain.com/en/latest/modules/memory/examples/adding_memory_chain_multiple_inputs.html
    # https://medium.com/@avra42/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842

    # st.sidebar.write('## Previous Chat History')
    history = st.session_state.memory.load_memory_variables({})
    # st.sidebar.write(history)

    # chain1: generate assumptions
    assumptions_chain = generate_assumption_chain()
    query_docs = vectordb.similarity_search(query_str)
    result =  assumptions_chain({
        "input_documents": query_docs, "question": query_str,'chat_history': history},
        return_only_outputs=False
    )
    st.sidebar.write('## Assumption')
    st.sidebar.write(result)

    # chain2: genetate possible answers with scores
    possible_answers_chain = generate_answer_chain()
    possible_answers = possible_answers_chain.run(result['output_text'])
    st.sidebar.write('## Asnswers')
    st.sidebar.write(possible_answers)

    # chain3 generate possible solutions based on highest scores
    final_answer_chain = generate_final_chain()
    query_docs = vectordb.similarity_search(possible_answers)
    final_result =  final_answer_chain({
        "input_documents": query_docs, "possible_answers": possible_answers,'chat_history': history},
        return_only_outputs=False
    )
    st.sidebar.write('## Final Answer')
    st.sidebar.write(final_result)

    # chain4: rephrase like a human
    human_response_chain = rephrase_like_a_human_chain()
    query_docs = vectordb.similarity_search('什麼是何明政的處理原則以及可能會有的回應風格？')
    human_resonse =  human_response_chain({
        "input_documents": query_docs, "final_answers": final_result['output_text'],'chat_history': history},
        return_only_outputs=False
    )
    st.sidebar.write('## AI Human Response')
    st.sidebar.write(human_resonse)


    # save all message to memory
    st.session_state.memory.chat_memory.add_user_message(query_str)
    st.session_state.memory.chat_memory.add_ai_message(final_result['output_text'])
    st.sidebar.write('## Current Chat History')
    history = st.session_state.memory.load_memory_variables({})
    st.sidebar.write(history)

    return human_resonse['output_text']

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


