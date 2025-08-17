# DeepScholar → Academic-focused assistant (ArXiv + PDFs)



A research-grade Streamlit application that integrates web information retrieval and document-grounded question answering into a unified conversational interface. Built with LangChain, powered by Groq’s low-latency LLMs, and equipped with state-of-the-art embedding and retrieval pipelines, this project demonstrates how to combine autonomous agents with retrieval-augmented generation (RAG) for practical, multimodal knowledge access.

✨ Core Capabilities
🧭 Web Search Agent

-> Tool-augmented reasoning via LangChain Agents.

-> Integrated sources: DuckDuckGo, Wikipedia, and ArXiv (scientific abstracts).

-> Automatic tool selection using ReAct (Reasoning + Acting) paradigm.

-> Fast, low-latency inference with Groq Llama3-8B-8192.

📄 PDF Conversational QA

-> Upload arbitrary PDF documents.

-> Document parsing with PyPDFLoader and hierarchical chunking (RecursiveCharacterTextSplitter).

-> Dense embeddings via HuggingFace (all-MiniLM-L6-v2).

-> FAISS vector store for high-performance similarity search.

-> Retrieval-augmented QA pipeline (RetrievalQA) for grounded responses.

@Future Extensions

-> Multi-document ingestion & cross-document reasoning.

-> Model selection: Llama3-70B, Mixtral, Gemma via Groq.

-> Enhanced retrieval strategies (MMR, hybrid lexical+dense).

-> Deployment on Streamlit Cloud or Docker for reproducibility.

👤 Author

Deepesh Singh
AI & Agentic Systems Builder | Bridging Research and Real-World Applications

🌐 LinkedIn

💻 GitHub
