import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG -----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
DOCS_FILE = "documents.pkl"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com"
}

# ----------------- UI SETUP -----------------
st.set_page_config(page_title="INSIGHT FLOW 🔎", layout="wide")
st.title("INSIGHT FLOW 🔎 : AI News Research Tool")
st.markdown("Fetch articles, process them, and ask AI-powered questions.")

st.sidebar.header("⚙️ Settings")
url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")
urls = [u.strip() for u in (url1, url2, url3) if u]

# ----------------- HELPERS -----------------
def extract_title(soup):
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "No Title"

def fetch_and_extract(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "aside"]):
        tag.decompose()

    title = extract_title(soup)
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    return title, text

def save_docs(docs):
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

def load_docs():
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "rb") as f:
            return pickle.load(f)
    return []

def clear_docs():
    if "docs" in st.session_state:
        del st.session_state["docs"]
    if os.path.exists(DOCS_FILE):
        os.remove(DOCS_FILE)
    st.sidebar.success("🗑️ Data cleared")

# ----------------- LOAD SAVED DATA -----------------
if "docs" not in st.session_state:
    saved = load_docs()
    if saved:
        st.session_state["docs"] = saved
        st.sidebar.info(f"Loaded {len(saved)} chunks")

# ----------------- FETCH & PROCESS -----------------
if st.sidebar.button("Fetch and Process Data"):
    if not urls:
        st.warning("Enter at least one URL")
    else:
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        for url in urls:
            try:
                title, text = fetch_and_extract(url)
                chunks = splitter.split_text(text)

                for i, chunk in enumerate(chunks, 1):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": url,
                                "title": title,
                                "chunk": i
                            }
                        )
                    )
                st.sidebar.success(f"Fetched {len(chunks)} chunks")
            except Exception as e:
                st.sidebar.error(str(e))

        st.session_state["docs"] = documents
        save_docs(documents)
        st.success(f"✅ Total {len(documents)} chunks ready")

# ----------------- CLEAN -----------------
if st.sidebar.button("Clean Data"):
    clear_docs()

# ----------------- RETRIEVAL -----------------
def retrieve_relevant_chunks(query, docs, k):
    texts = [d.page_content for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [docs[i] for i in top_indices]

def ask_groq(question, context):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer strictly from the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )
    return response.choices[0].message.content

# ----------------- QUERY UI -----------------
query_text = st.text_area(
    "🔍 Ask your questions (one per line):",
    placeholder="What is the outlook for Tata Motors?"
)

top_k = st.slider("Top relevant chunks", 1, 5, 3)

# ----------------- RUN QUERY -----------------
if st.button("Get Summaries") and query_text.strip():
    if "docs" not in st.session_state:
        st.error("No data available. Fetch articles first.")
    else:
        docs = st.session_state["docs"]
        queries = [q.strip() for q in query_text.split("\n") if q.strip()]

        st.subheader("📝 AI Answers")

        for q in queries:
            relevant_docs = retrieve_relevant_chunks(q, docs, top_k)
            context = "\n".join(d.page_content for d in relevant_docs)
            answer = ask_groq(q, context)

            st.markdown(f"### ❓ {q}")
            st.success(answer)
            st.divider()
        st.sidebar.success("✅ Queries processed")