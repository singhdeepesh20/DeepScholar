import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

st.set_page_config(page_title="DeepSearch – Intelligent Web & Document Exploration Agent", layout="wide")

st.title("DeepSearch – Intelligent Web & Document Exploration Agent")

huggingface_api_key = st.secrets.get("HUGGINGFACE_API_KEY", None)
if not huggingface_api_key:
    st.error("Hugging Face API key not found in Streamlit secrets. Please add it to your `secrets.toml`.")
    st.stop()

groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=huggingface_api_key)

source_type = st.selectbox("Choose data source", ["Webpage", "PDF"])
docs = []

if source_type == "Webpage":
    url = st.text_input("Enter webpage URL")
    if st.button("Load Webpage") and url:
        loader = WebBaseLoader(url)
        docs = loader.load()

elif source_type == "PDF":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="mixtral-8x7b-32768", api_key=groq_api_key)

    prompt = ChatPromptTemplate.from_template(
        "You are DeepSearch, an intelligent assistant. Use the provided context to answer.\n\nContext:\n{context}\n\nQuestion: {input}"
    )

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    query = st.text_input("Enter your search query:")
    if st.button("Search") and query:
        try:
            result = chain.invoke({"input": query})
            st.subheader("Answer")
            st.write(result["answer"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
