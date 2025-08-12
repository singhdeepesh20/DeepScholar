import os
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory



# ========================
HF_TOKEN = st.secrets["HF_TOKEN"]  # Hugging Face API Key from Streamlit Secrets



# ========================
st.sidebar.title("Settings")
mode = st.sidebar.radio("Choose Mode:", ["Search Mode", "PDF Mode"])
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API Key to continue.")



# ========================
if "search_memory" not in st.session_state:
    st.session_state.search_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "pdf_memory" not in st.session_state:
    st.session_state.pdf_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



# ========================
def get_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )



# ========================
def search_mode():
    st.title("DeepSearch – Intelligent web & document exploration agent")


    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    search = DuckDuckGoSearchRun(name="Search")
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

    llm = get_llm()
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=st.session_state.search_memory,
        handle_parsing_errors=True
    )

    for msg in st.session_state.search_memory.chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.search_memory.chat_memory.add_user_message(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.write(response)
        st.session_state.search_memory.chat_memory.add_ai_message(response)

# ========================
def pdf_mode():
    st.title("📄 PDF Q&A")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=".", use_auth_token=HF_TOKEN)
        vectorstore = FAISS.from_documents(pages, embeddings)
        retriever = vectorstore.as_retriever()

        llm = get_llm()
        pdf_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            memory=st.session_state.pdf_memory
        )

        for msg in st.session_state.pdf_memory.chat_memory.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input("Ask something about the PDF..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.pdf_memory.chat_memory.add_user_message(prompt)

            with st.chat_message("assistant"):
                result = pdf_chain({"question": prompt})
                answer = result["answer"]
                st.write(answer)
            st.session_state.pdf_memory.chat_memory.add_ai_message(answer)


# ========================
if groq_api_key:
    if mode == "Search Mode":
        search_mode()
    elif mode == "PDF Mode":
        pdf_mode()
else:
    st.warning("Please enter your Groq API Key in the sidebar.")
