iimport os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun

# ----------------------------
# 1. Load Secrets from Streamlit
# ----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if not HF_TOKEN:
    st.error("Please set `HF_TOKEN` in Streamlit Secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ----------------------------
# 2. App Title
# ----------------------------
st.title("🔍 DeepQuery.AI")
st.write("Search online + PDF knowledge in one agent.")

# ----------------------------
# 3. Upload PDF
# ----------------------------
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
pdf_vectorstore = None

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"use_auth_token": HF_TOKEN}
    )

    pdf_vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("✅ PDF loaded & indexed.")

# ----------------------------
# 4. Create Tools
# ----------------------------
tools = []

# PDF QA Tool
if pdf_vectorstore:
    pdf_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 3})
    pdf_qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        ),
        retriever=pdf_retriever
    )
    tools.append(
        Tool(
            name="PDF QA",
            func=pdf_qa_chain.run,
            description="Use this tool to answer questions from the uploaded PDF."
        )
    )

# Web Search Tool
search_tool = DuckDuckGoSearchRun()
tools.append(
    Tool(
        name="Web Search",
        func=search_tool.run,
        description="Use this tool to search the internet."
    )
)

# ----------------------------
# 5. Initialize Agent
# ----------------------------
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# ----------------------------
# 6. User Query
# ----------------------------
query = st.text_input("💬 Ask me anything...")
if query:
    with st.spinner("Thinking..."):
        response = agent.run(query)
    st.write("**Answer:**", response)
