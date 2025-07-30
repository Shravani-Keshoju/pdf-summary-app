import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import textwrap
import time
import os
import streamlit.components.v1 as components

# Set your Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

generation_model = genai.GenerativeModel("models/gemini-2.5-pro")
embedding_model = "models/embedding-001"
# Initialize page layout
st.set_page_config(layout="wide")
st.markdown("<h1>📄 PDF Auto-Summarizer with Gemini</h1>", unsafe_allow_html=True)

# Initialize terminal logs
if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {msg}")
    # Keep only the last 50 logs
    st.session_state.logs = st.session_state.logs[-50:]

def render_terminal():
    logs_text = "\n".join(st.session_state.logs)
    styled_terminal = f"""
    <div style='position: fixed; top: 80px; right: 10px; width: 30vw; height: 80vh;
                background-color: #000; color: #0f0; font-family: monospace;
                font-size: 12px; overflow-y: auto; border-radius: 8px; padding: 10px;
                z-index: 999; box-shadow: 0 0 10px rgba(0,255,0,0.3);'>
        <pre>{logs_text}</pre>
    </div>
    """
    components.html(styled_terminal, height=800)

def extract_text_from_pdf(uploaded_file):
    log("📥 Extracting text from PDF...")
    reader = PdfReader(uploaded_file)
    text = ""
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            log(f"📄 Page {i+1}: extracted {len(content)} characters.")
            text += content + "\n"
        else:
            log(f"⚠️ Page {i+1}: no extractable text.")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    log("✂️ Chunking text...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    log(f"✅ Created {len(chunks)} chunks.")
    return chunks

def embed_chunks(chunks):
    log(f"🧠 Generating embeddings for {len(chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        log(f"🔄 Embedding chunk {i+1}/{len(chunks)}...")
        response = genai.embed_content(
            model=embedding_model,
            content=chunk,
            task_type="RETRIEVAL_DOCUMENT",
            title=f"chunk_{i+1}"
        )
        vector = response["embedding"]
        log(f"✅ Embedded chunk {i+1} | Vector preview: {vector[:5]}")
        embeddings.append((chunk, vector))
        time.sleep(0.25)
    return embeddings

def generate_summary(chunks):
    log("🧾 Generating summary with Gemini-Pro...")
    prompt = (
        "You are a document summarization expert. Read the following content and summarize clearly:\n\n"
        + "\n\n".join(chunks)
    )
    response = generation_model.generate_content(prompt)
    log("✅ Summary generated.")
    return response.text.strip()

# UI - Left Section (main content)
with st.container():
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        _ = embed_chunks(chunks)
        summary = generate_summary(chunks)
        st.markdown("### 📃 Summary")
        st.markdown(summary)

# Render sticky terminal
render_terminal()
