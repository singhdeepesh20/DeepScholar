import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -------------------------------
# ✅ 1. SET YOUR HUGGING FACE API KEY HERE (LOCAL TESTING)
# -------------------------------
HUGGINGFACE_API_KEY = "hf_your_actual_api_key_here"  # Replace with your HF token

# Optional: Use Streamlit secrets if available
# HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", HUGGINGFACE_API_KEY)

if not HUGGINGFACE_API_KEY:
    st.error("Hugging Face API key is missing. Please set it in the script or in Streamlit secrets.")
    st.stop()

# -------------------------------
# ✅ 2. Streamlit UI
# -------------------------------
st.set_page_config(page_title="DeepSearch – Intelligent Web & Document Exploration Agent", layout="wide")
st.title("🔍 DeepSearch – Intelligent Web & Document Exploration Agent")

uploaded_file = st.file_uploader("📄 Upload PDF or TXT file", type=["pdf", "txt"])
web_url = st.text_input("🌐 Enter a web page URL to load")

query = st.text_input("💬 Ask a question about the document/web page")

# -------------------------------
# ✅ 3. Load and split documents
# -------------------------------
documents = []

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name)
    
    documents.extend(loader.load())

if web_url:
    url_loader = UnstructuredURLLoader(urls=[web_url])
    documents.extend(url_loader.load())

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # -------------------------------
    # ✅ 4. Create embeddings & FAISS index
    # -------------------------------
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    retriever = vector_store.as_retriever()

    # -------------------------------
    # ✅ 5. Load Hugging Face LLM
    # -------------------------------
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # -------------------------------
    # ✅ 6. Handle queries
    # -------------------------------
    if query:
        with st.spinner("🤖 Thinking..."):
            result = qa_chain({"query": query})
        
        st.subheader("📌 Answer:")
        st.write(result["result"])

        with st.expander("📂 Source Documents"):
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata}")
                st.write(doc.page_content[:500] + "...")
else:
    st.info("📢 Please upload a document or enter a URL to start.")
