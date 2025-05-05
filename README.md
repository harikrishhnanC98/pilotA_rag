# PDF → RAG Chunking & Retrieval Pilot

This project implements a Retrieval-Augmented Generation (RAG) pipeline to chunk a 130-page German radiology guideline PDF into searchable passages. Built in Google Colab (free tier), it focuses on semantic chunking for retrieval.

## Project Structure

- **pilotA_rag.ipynb**: Main notebook containing the backend code for segmentation, RAG pipeline logic, and results.
- **rag_pipeline.py**: Python file version of the segmentation and RAG logic, used to support the UI(not required to run separately).
- **app.py**: Streamlit script for the web-based user interface.
- **pilotA_rag_ui.ipynb**: Notebook for running and displaying the Mini-UI (using ipywidgets or Streamlit setup).
- **streamlit_ui.png** : Screenshot from the streamlit UI containing the querying part and results.

## Setup Instructions

### Environment
- Google Colab (free tier, no GPU used/required).

### Data
- **PDF**: `Guideline_atraumatische_Femurkopfnekrose_2019-09_1-abgelaufen.pdf`.
- **JSON**: `questions_answers.json` (test queries + answers).

### Storage
- Store in `/content/drive/MyDrive/rag_project/`.

### Installation
1. **Mount Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Install dependencies**:
    ```bash
    pip install PyMuPDF rapidfuzz sentence-transformers faiss-cpu numpy streamlit pyngrok
    ```

### Run the Pipeline
- Run `pilotA_rag.ipynb` to generate `segments_headings.json` and compute results.
- Run `app.py` and `pilotA_rag_ui.ipynb` for the streamlit UI.


## Dependencies
- **PyMuPDF**: PDF parsing.
- **rapidfuzz**: Title/question matching.
- **sentence-transformers**: Embeddings (deepset/gbert-base).
- **faiss-cpu**: Vector indexing.
- **numpy**: Array operations.
- **streamlit**: Web framework for UI.
- **pyngrok**: Tunneling for public access.

## Chunking Logic

The pipeline uses a simple title-based heuristic to create topic-coherent coarse chunks, prioritizing “FRAGENKATALOG” (pages 19–26). The heuristic is based on the assumption of having coherent topics within their respective title sections(similar to the reconstruction of metadata/TOC useful for segmentation). Based on observation all the titles behave uniformly with capitalized format making it easy to extract them. Using the detected headings, segments are created. The coarse chunks are then segmented further into paragraph levels to preserve the topic coherence within the title-based (coarse) sections.

- **Heading Extraction**: Detects capitalized headings (regex: `^[A-ZÄÖÜ0-9\s,.:;()-]+$`), filtering noise (e.g., references).
- **Segmentation**: Groups text under headings, skipping pages 1–3 and ≥106. Identifies “FRAGENKATALOG” by title or pages 19–26.
- **Question Detection**: Uses regex (`^(?:\d+.\s*|\[A-Z]\[.)]\s*(WIE|WELCHE|...)\s*.+?`) for questions (e.g., “Welche bildgebende Methode...?”).
- **Chunking**: Splits into 80–500 words (500 for “FRAGENKATALOG”), keeping paragraphs (marked by `|||PARAGRAPH|||`). Merges small chunks (<80 words) on the same page.

### Reason
- Aligns with document structure, ensuring “FRAGENKATALOG” question-answer pairs (e.g., “MRT ist der Goldstandard”) are retrievable.

## Improvement Suggestion (TODO)

### Meta-Chunking
The current approach even though outputs decent recall and similarity scores, precision remains low indicating the presence of irrelevant text inside the chunks along with words from gold answer. Therefore better chunking strategies especially based on recent research would be extremely helpful. Implement the Meta-Chunking approach from “Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception” (or similar), which uses a language model to split text at topic shifts. This could better isolate “FRAGENKATALOG” question-answer pairs.

### Status
- Attempted but failed due to torchvision errors in Colab.

### Next Steps
- Fix the dependencies and suitable stack/models to implement Meta-chunking.
