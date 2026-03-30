<div align="center">

# DeepScholar — Academic AI Assistant

### Research-Grade Conversational System for Web & Document Intelligence

</div>

---

## Overview

**DeepScholar** is a research-oriented AI assistant designed for **academic and knowledge-intensive workflows**, combining **web-scale information retrieval** with **document-grounded reasoning** in a unified conversational interface.

Built using **LangChain agents, Retrieval-Augmented Generation (RAG), and Groq-powered LLMs**, the system demonstrates how modern AI architectures can enable **reliable, context-aware, and multi-source knowledge access**.

This project reflects a strong focus on **agentic AI systems, retrieval pipelines, and real-world applicability of LLMs**.

---

## Core Capabilities

### 🧭 Web Search Agent

* Tool-augmented reasoning using **LangChain Agents**
* Integrated knowledge sources:

  * DuckDuckGo (general web search)
  * Wikipedia (structured knowledge)
  * ArXiv (scientific literature)
* **ReAct (Reasoning + Acting)** paradigm for dynamic tool selection
* Low-latency inference via **Groq (LLaMA3-8B-8192)**

### 📄 PDF Conversational QA

* Upload and analyze arbitrary PDF documents
* Document parsing via **PyPDFLoader**
* Hierarchical chunking using **RecursiveCharacterTextSplitter**
* Dense embeddings with **HuggingFace (all-MiniLM-L6-v2)**
* High-performance retrieval using **FAISS vector store**
* Context-grounded responses using **RetrievalQA pipeline**

---

## System Architecture

```
User Query
   ↓
Agent Router (LangChain)
   ↓
[ Web Tools ]        [ PDF Retrieval ]
(DuckDuckGo,        (FAISS + Embeddings)
 Wikipedia, ArXiv)
   ↓                      ↓
        Context Aggregation
                ↓
          Groq LLM (LLaMA3)
                ↓
        Final Response
```

---

## Tech Stack

* **LLM**: Groq (LLaMA3-8B-8192)
* **Framework**: LangChain (Agents + Chains)
* **Embeddings**: HuggingFace (SentenceTransformers)
* **Vector Store**: FAISS
* **Document Processing**: PyPDFLoader
* **Frontend**: Streamlit

---

## Engineering Approach

### Agentic AI Design

* Autonomous decision-making via tool selection
* ReAct-based reasoning for dynamic workflows

### Retrieval-Augmented Generation (RAG)

* Grounded responses using document context
* Reduced hallucination through retrieval pipelines

### Hybrid Knowledge Access

* Combines **live web data** with **private document understanding**
* Enables both exploratory research and targeted Q&A

---

## What This Project Demonstrates

* Integration of **agents + RAG in a unified system**
* Real-world application of **multisource retrieval pipelines**
* Strong understanding of **LLM orchestration and reasoning frameworks**
* Ability to design **research-grade AI systems with practical usability**

---

## Future Extensions

* Multi-document ingestion & cross-document reasoning
* Support for advanced models (LLaMA3-70B, Mixtral, Gemma via Groq)
* Hybrid retrieval (dense + lexical, MMR optimization)
* Deployment via **Streamlit Cloud / Docker** for reproducibility

---

## Author

**Deepesh Singh**
AI & Agentic Systems Builder | Bridging Research and Real-World Applications

🌐 LinkedIn: [https://www.linkedin.com/in/contactdeepeshsingh/](https://www.linkedin.com/in/contactdeepeshsingh/)
💻 GitHub: [https://github.com/singhdeepesh20](https://github.com/singhdeepesh20)

---

<div align="center">

### "From information retrieval to intelligent reasoning systems."

<

