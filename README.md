# Retrieval-Augmented Answers

Answers questions from my own documents instead of model's memory.

## What it includes
- Documents chunked and embedded (500 char chunks, TF-IDF embeddings)
- Retrieved passages shown with the answer (with source + score)
- Honest 'not in the documents' response if answer missing

## How to run
pip install -r requirements.txt
streamlit run app.py

## Example
Q: Library timing?
A: Shows passage from documents/sample.txt with citation.
Q: Who is president?
A: This information is not in the provided documents.
