# =====================================================================
# 🤖 Streamlit RAG Assistant with Semantic Cache + Context Memory
# ---------------------------------------------------------------------
# Includes:
#   - PDF Upload & Chunking
#   - Embedding + Chroma VectorDB
#   - Semantic Caching
#   - Conversational Context
#   - LLM (LangChain-Groq: Llama-3.3-70B)
# =====================================================================

__import__('pysqlite3')                                                 # Dynamically imports pysqlite3 (a standalone SQLite module that supports modern features).
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')                   # Replaces the default sqlite3 module with pysqlite3. This is helpful in environments like Streamlit Cloud where SQLite might not support fulltext search (FTS5) or other features used in vector databases.

# ==============================
# 🧱 1️⃣ IMPORTS & SETUP
# ==============================
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


import streamlit as st
import os, time, tempfile
from datetime import datetime
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="RAG Assistant", layout="wide")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0


# ==============================
# 🔐 Load Groq API Key (Simple & Flexible)
# ==============================

api_key = os.getenv("GROQ_API_KEY")         # Enable while using on local machine

# api_key = st.secrets["GROQ_API_KEY"]        # Enable while deploying in streamlit

# 2️⃣ Validate
if not api_key:
    st.error("❌ No GROQ_API_KEY found. Please add it in `.env`, Streamlit Secrets, or sidebar input.")
    st.stop()

# ==============================
# ⚙️ 2️⃣ CONSTANTS & MODEL SETUP
# ==============================
EMBED_MODEL = "all-MiniLM-L6-v2"
CACHE_THRESHOLD = 0.88

# Embeddings
# embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)      # modified after the meta tensor error occured
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Persistent Vector DB & Semantic Cache
vector_db = Chroma(
    collection_name="pdf_docs",
    embedding_function=embeddings,
    persist_directory="./vectordb"
)
cache_db = Chroma(
    collection_name="semantic_cache",
    embedding_function=embeddings,
    persist_directory="./cache"
)

# LLM (LangChain-Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=500,
    api_key=api_key
)

# ==============================
# 📄 3️⃣ HELPER FUNCTIONS
# ==============================

def load_and_split_pdf(pdf_path: str):
    """Extract and split PDF content into manageable chunks."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def semantic_cache_get(query: str):
    """Retrieve cached answer if similar query exists in cache."""
    hits = cache_db.similarity_search_with_relevance_scores(query, k=1)
    if hits:
        doc, score = hits[0]
        if score >= CACHE_THRESHOLD:
            st.session_state.cache_hits += 1
            return doc.metadata.get("response")
    st.session_state.cache_misses += 1
    return None


# def semantic_cache_set(query: str, response: str):
#     """Store query-response pair into semantic cache."""
#     cache_db.add_texts(
#         [query],
#         metadatas=[{
#             "response": response,
#             "timestamp": datetime.utcnow().isoformat()
#         }]
#     )
#     cache_db.persist()

def semantic_cache_set(query: str, response):
    """Store query-response pair in semantic cache, handling complex LLM metadata."""
    # If response is an AIMessage, extract its text content
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)

    cache_db.add_texts(
        [query],
        metadatas=[{
            "response": response_text,  # Only save plain text
            "timestamp": datetime.utcnow().isoformat()
        }]
    )
    cache_db.persist()



def append_session_context(user_query, assistant_response):
    """Add the latest Q&A turn into session history."""
    st.session_state.history.append({
        "query": user_query,
        "response": assistant_response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


def build_prompt_with_context(docs, history, query):
    """Construct prompt combining chat history + document context."""
    chat_context = "\n".join([
        f"User: {turn['query']}\nAssistant: {turn['response']}"
        for turn in history[-3:]
    ])
    doc_context = "\n".join([d.page_content for d in docs])
    return f"""
You are an intelligent assistant that answers based on the provided
document context and recent conversation.

Chat History:
{chat_context}

Document Context:
{doc_context}

User Question:
{query}

Respond concisely, factually, and reference document insights when relevant.
""".strip()


# ==============================
# 🧩 4️⃣ STREAMLIT FRONTEND
# ==============================
st.title("🤖 RAG Assistant with Semantic Cache + Conversational Context")

# --- Upload PDFs ---
with st.expander("📄 Step 1: Upload and Ingest PDFs"):
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("📥 Process & Store PDFs"):
        if uploaded_files:
            total_chunks = 0
            with st.spinner("Extracting text and creating embeddings..."):
                for pdf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf.getbuffer())
                        tmp_path = tmp.name

                    chunks = load_and_split_pdf(tmp_path)
                    vector_db.add_documents(chunks)
                    vector_db.persist()
                    total_chunks += len(chunks)
            st.success(f"✅ Successfully added {total_chunks} text chunks to Vector Database!")
        else:
            st.warning("Please upload at least one PDF to continue.")

# --- Ask Questions ---
st.subheader("💬 Step 2: Ask Questions from Your PDFs")

query = st.text_input("Type your question here:")
if st.button("Ask Question"):
    if query.strip():
        start = time.time()

        # Step 1 - Check semantic cache
        cached_response = semantic_cache_get(query)
        if cached_response:
            st.info("🧠 Retrieved from Semantic Cache")
            response = cached_response
        else:
            # Step 2 - Retrieve similar content from Vector DB
            retriever = vector_db.as_retriever(search_kwargs={"k": 4})
            relevant_docs = retriever.get_relevant_documents(query)

            # Step 3 - Build prompt
            prompt = build_prompt_with_context(relevant_docs, st.session_state.history, query)

            # Step 4 - LLM Response
            with st.spinner("💭 Generating intelligent response..."):
                response = llm.invoke(prompt)

            # Step 5 - Save to cache
            semantic_cache_set(query, response)

        # Step 6 - Save to session
        append_session_context(query, response)

        duration = round(time.time() - start, 2)
        st.success(response)
        st.caption(f"⏱️ Time Taken: {duration}s")
    else:
        st.warning("Please enter a valid question before submitting.")

# --- Conversation History ---
st.divider()
st.subheader("🧠 Recent Conversation History (Last 5 Turns)")

for turn in st.session_state.history[-5:][::-1]:
    st.markdown(f"**🧑 User:** {turn['query']}")
    st.markdown(f"**🤖 Assistant:** {turn['response']}")
    st.caption(f"🕒 {turn['timestamp']}")
    st.write("---")

# --- Sidebar Metrics ---
st.sidebar.header("⚙️ System Performance Metrics")
st.sidebar.metric("Cache Hits", st.session_state.cache_hits)
st.sidebar.metric("Cache Misses", st.session_state.cache_misses)
st.sidebar.metric("Total Queries", len(st.session_state.history))
st.sidebar.write("Semantic Cache Threshold:", CACHE_THRESHOLD)
st.sidebar.caption("Higher = stricter cache similarity requirement.")

# --- Reset Session History ---
with st.expander("🔄 Reset Session History"):
    if st.button("Reset History"):
        st.session_state.history = []
        st.session_state.cache_hits = 0

        st.session_state.cache_misses = 0


