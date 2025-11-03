# 🤖 Streamlit RAG Assistant with Semantic Cache + Context Memory

A powerful and interactive **Retrieval-Augmented Generation (RAG)** assistant built using **Streamlit**, **LangChain**, **Groq (Llama-3.3-70B)**, and **ChromaDB**.  
This application enables users to upload PDF documents, query them conversationally, and receive intelligent responses grounded in the document context — with **semantic caching** to speed up repeated or similar queries.

---

## 🧩 Features

| Feature | Description |
|----------|--------------|
| 📄 **PDF Upload & Chunking** | Upload one or more PDF files and automatically extract and split their content into chunks. |
| 🔍 **Embedding + Vector Store (Chroma)** | Text chunks are embedded using HuggingFace MiniLM and stored in a persistent ChromaDB. |
| 🧠 **Semantic Caching** | Queries and responses are cached based on similarity to minimize redundant LLM calls. |
| 💬 **Conversational Context Memory** | Keeps track of the last few question-answer turns for contextual understanding. |
| ⚙️ **Groq LLM Integration** | Uses the **Llama-3.3-70B** model hosted by Groq for lightning-fast, factual responses. |
| 📊 **Session Metrics Dashboard** | Sidebar shows cache hits, misses, and total queries. |
| 💾 **Persistent Storage** | Both VectorDB and Cache persist between sessions using local directories. |

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/rag-streamlit-assistant.git
cd rag-streamlit-assistant
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Environment Variables
Create a `.env` file in the project root and add:
```bash
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run rag_streamlit_app.py
```

---

## 📊 Sidebar Metrics Explained

| Metric | Description |
|---------|-------------|
| **Cache Hits** | Number of responses retrieved from semantic cache. |
| **Cache Misses** | Number of times the LLM was invoked. |
| **Total Queries** | Total questions asked in the session. |
| **Threshold** | Similarity cutoff (default = 0.88). |

---

🔍 Example Usage

1. Upload a company policy PDF.
2. Ask

```  
What are the duties of the employer according to this document?
```
The app retrieves, interprets, and summarizes the relevant section using contextual understanding.
---
⚡ Key Advantages

🚀 Faster responses due to semantic caching

🧩 Accurate document-based retrieval

💬 Context-aware multi-turn chat

💾 Persistent storage for reuse

🔒 Secure API key handling via .env
---
⚠️ Limitations

ChromaDB stores data locally (for production, connect to a remote DB).

Response quality depends on PDF text quality and chunking size.

No user authentication yet (for internal or prototype use).

----
🧭 Future Enhancements

Add colored sidebar LLM performance cards (model, tokens, time).

Integrate OpenAI or Anthropic APIs for hybrid LLM comparison.

Introduce multi-user session management.

Add feedback collection for answer accuracy.
---

## 👨‍💻 Author
**Murali Krishna Reddy B**  
*Data Analytics & Generative AI Engineer*  

**https://www.linkedin.com/in/muralikrishnareddyb** 

**https://github.com/reachmurali2** 

📧 Contact: reachmurali2@gmail.com
