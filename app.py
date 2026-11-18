import streamlit as st
import os, glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG - My Docs Only")
st.title("Retrieval-Augmented Answers")
st.caption("Answers ONLY from your own documents")

# 1. Chunking function
def chunk_text(text, size=500, overlap=100):
    chunks=[]
    for i in range(0,len(text),size-overlap):
        chunks.append(text[i:i+size])
    return chunks

# Load docs
docs_path = "documents"
os.makedirs(docs_path, exist_ok=True)

all_chunks=[]
metas=[]
for file in glob.glob(f"{docs_path}/*.*"):
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            text=f.read()
            for idx, ch in enumerate(chunk_text(text)):
                all_chunks.append(ch)
                metas.append({"source": file, "chunk_id": idx})
    except: pass

if not all_chunks:
    st.warning("Pehle documents/ folder me 1 txt file banao. Example neeche hai.")
    st.stop()

# 2. Embedding with TF-IDF (lightweight, no API key needed)
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(all_chunks)

# UI
q = st.text_input("Apna sawal pucho:")
if q:
    q_vec = vectorizer.transform([q])
    scores = cosine_similarity(q_vec, doc_vectors)[0]
    top_idx = scores.argsort()[-3:][::-1]

    # 3. Honest check
    if scores[top_idx[0]] < 0.15:
        st.error("This information is not in the provided documents.")
    else:
        st.subheader("Answer (from docs only):")
        st.write(all_chunks[top_idx[0]][:800])

        st.subheader("Retrieved passages shown with the answer:")
        for i in top_idx:
            st.info(f"**Source:** {metas[i]['source']} | Chunk {metas[i]['chunk_id']} | Score {scores[i]:.2f}\n\n{all_chunks[i][:500]}")
