import streamlit as st
import json
import os
import importlib.util

# Set working directory
rag_project_path = "/content/drive/MyDrive/rag_project"
os.chdir(rag_project_path)

# Dynamically import rag_pipeline
rag_pipeline_path = os.path.join(rag_project_path, "rag_pipeline.py")
if not os.path.exists(rag_pipeline_path):
    st.error(f"rag_pipeline.py not found in {rag_project_path}")
    st.stop()

spec = importlib.util.spec_from_file_location("rag_pipeline", rag_pipeline_path)
rag_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_pipeline)
embed_and_index = rag_pipeline.embed_and_index
search = rag_pipeline.search

# Set page title
st.title("RAG Pipeline for Radiology Guidelines")

# Load data
pdf_path = os.path.join(rag_project_path, "Guideline_atraumatische_Femurkopfnekrose_2019-09_1-abgelaufen.pdf")
eval_json_path = os.path.join(rag_project_path, "questions_answers.json")
if not os.path.exists(pdf_path):
    st.error("PDF not found! Check path: " + pdf_path)
    st.stop()

# Load evaluation questions for dropdown
try:
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    eval_questions = [item['question'] for item in eval_data]
except FileNotFoundError:
    st.error("questions_answers.json not found! Check path: " + eval_json_path)
    st.stop()

# Initialize model and index
@st.cache_resource
def load_index():
    model, index, segments = embed_and_index(final_segments)
    return model, index, segments

# Load segments
try:
    with open(os.path.join(rag_project_path, 'segments_headings.json'), 'r', encoding='utf-8') as f:
        final_segments = json.load(f)
    model, index, segments = load_index()
except FileNotFoundError:
    st.error("Run rag_pipeline.py first to generate segments_headings.json")
    st.stop()

# Query input with dropdown
query = st.selectbox("Select a query:", eval_questions, index=0)

# Search and display results
if query:
    st.subheader("Top-3 Retrieved Chunks")
    results = search(query, model, index, segments, k=3)
    for i, res in enumerate(results, 1):
        st.write(f"**Chunk {i}**")
        st.write(f"**Page:** {res['page']}")
        st.write(f"**Score:** {res['score']:.4f}")
        st.write(f"**Text:** {res['text'][:500]}...")
        st.write("---")
    # Precision and recall
    st.subheader("Metrics")
    st.write(f"**Average Precision@3:** 0.0093")
    st.write(f"**Average Recall@3:** 0.6088")
else:
    st.info("Please select a query to see results.")