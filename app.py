import os
import tempfile
import streamlit as st
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="AI Search & PDF Chat")

st.title("DeepScholar - AI Search & PDF Chat")
st.write("Select a mode from the sidebar: **Web Search** or **PDF Chat**")


st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
mode = st.sidebar.radio("Select Mode:", ["Web Search", "PDF Chat"])


if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()


llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)


if mode == "Web Search":
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    search = DuckDuckGoSearchRun(name="Search")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    if prompt := st.chat_input(placeholder="Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        tools = [search, arxiv, wiki]
        search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                        handle_parsing_errors=True)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


elif mode == "PDF Chat":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name


            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            retriever = vectorstore.as_retriever()

            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        if "pdf_messages" not in st.session_state:
            st.session_state["pdf_messages"] = [
                {"role": "assistant", "content": "PDF uploaded! Ask me questions about it."}
            ]

        for msg in st.session_state.pdf_messages:
            st.chat_message(msg["role"]).write(msg['content'])

        if question := st.chat_input(placeholder="Ask a question about the PDF..."):
            st.session_state.pdf_messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            with st.chat_message("assistant"):
                answer = qa_chain.run(question)
                st.session_state.pdf_messages.append({"role": "assistant", "content": answer})
                st.write(answer)
