import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate



# -----------------------
st.set_page_config(page_title="DeepSearch – Intelligent web & document exploration agent", layout="wide")
st.title(" DeepSearch – Intelligent web & document exploration agent")



# -----------------------
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]



# -----------------------
groq_api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()



# -----------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0
)



# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    token=HF_KEY
)



# -----------------------
if "search_memory" not in st.session_state:
    st.session_state.search_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "pdf_memory" not in st.session_state:
    st.session_state.pdf_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



# -----------------------
mode = st.radio("Choose mode:", ["Search Mode", "PDF Mode"])



# -----------------------
if mode == "Search Mode":
    st.subheader(" Web Search Agent")

    def web_search_tool(query):
        # In real use, integrate an API or scraper here
        return f"Simulated search results for: {query}"

    tools = [
        Tool(
            name="WebSearch",
            func=web_search_tool,
            description="Search the internet for information"
        )
    ]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,  
        memory=st.session_state.search_memory,
        handle_parsing_errors=True,
        verbose=True
    )

    query = st.text_input("Ask me anything:")
    if st.button("Run Search") and query:
        try:
            response = search_agent.run(query)
        except Exception as e:
            response = f"⚠ Search failed: {e}"
        st.write(response)



# -----------------------
elif mode == "PDF Mode":
    st.subheader("DocuMind")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            memory=st.session_state.pdf_memory
        )

        pdf_query = st.text_input("Ask a question about the PDF:")
        if st.button("Run PDF Q&A") and pdf_query:
            try:
                response = qa_chain.run(pdf_query)
            except Exception as e:
                response = f"⚠️ PDF Q&A failed: {e}"
            st.write(response)
