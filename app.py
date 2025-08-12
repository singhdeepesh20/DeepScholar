import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType

# ------------------ Load Environment Variables ------------------
load_dotenv()

# Prefer Streamlit Secrets if deployed
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
if not HF_TOKEN:
    st.error("❌ Please set HF_TOKEN in Streamlit secrets or .env file.")
    st.stop()

# Required for HuggingFaceEmbeddings
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ------------------ Streamlit App ------------------
st.title("DeepSearch – Intelligent web & document exploration agent")
st.write("Ask questions using both the internet and your uploaded documents.")

# Groq API Key
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
if not api_key:
    st.warning("Enter Groq API key to continue.")
    st.stop()

# LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Online search tools
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
duck_tool = DuckDuckGoSearchRun(name="Search")

tools = [duck_tool, arxiv_tool, wiki_tool]

# PDF Upload
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)
        os.remove(temp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    def pdf_search_tool(query):
        results = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in results])

    pdf_tool = Tool(
        name="PDF Search",
        func=pdf_search_tool,
        description="Use this tool to answer questions based on uploaded PDF files."
    )

    tools.append(pdf_tool)

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user prompt
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"Error: {e}"
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
