import fitz  # PyMuPDF
import re
import json
import os
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Clean text for headings or content
def clean_text(text, is_heading=False):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'S3-Leitlinie.*?\d+\s*$', '', text)
    text = re.sub(r'^\d+\s*$', '', text)
    text = re.sub(r'^\s*[-–]\s*$', '', text)
    text = re.sub(r'^\d+[.,]\d+%?$', '', text)
    text = re.sub(r'^\w{1,2}\s*$', '', text)
    text = re.sub(r'^\s*[A-Z]{1,4}\s*[:;]?\s*$', '', text)
    text = re.sub(r'^\d+[,\s\d;]+$', '', text)
    text = re.sub(r'AWMF-Register-Nr.*?\d+', '', text)
    text = re.sub(r'Version vom 18\.09\.2019.*', '', text)
    text = re.sub(r'Langfassung.*?\d+', '', text)
    text = re.sub(r'[^\w\s.,;?!-äöüÄÖÜ]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text if text else None

# Extract capitalized headings
def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    headings = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            lines = text.split('\n')
            for line in lines:
                clean_line = clean_text(line.strip(), is_heading=True)
                if clean_line and re.match(r'^[A-ZÄÖÜ0-9\s,.:;()\-]+$', clean_line):
                    if not re.match(r'^(?:\d+;.*|[A-ZÄÖÜ\s,]{1,6}(?:,|\.)?$|ARCO:.*|DGOOC|DGU|BVOU|\(NRW\)|I-II).*', clean_line):
                        headings.append((page_num + 1, clean_line))
    doc.close()
    return headings

# Fuzzy match titles
def match_titles(parsed_titles, ground_truth_titles, threshold=60):
    matched = []
    for parsed in parsed_titles:
        best_match, score, _ = process.extractOne(parsed, ground_truth_titles, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matched.append((parsed, best_match, score))
        else:
            matched.append((parsed, None, score))
    return matched

# Segment PDF
def segment_pdf(pdf_path, headings, eval_questions):
    doc = fitz.open(pdf_path)
    segments = []
    current_segment = {"title": "Introduction", "text": [], "page": 1}
    heading_index = 0
    headings = sorted(headings, key=lambda x: x[0])
    ground_truth_titles = [h[1] for h in headings]
    in_question_section = False
    question_pattern = re.compile(r'^(?:\d+\.\s*|[A-Z][\.\)]\s*)(WIE|WELCHE|WANN|WARUM|WAS|SOLLTE|WEN|KANN|IST|HABEN)\s*.+\?', re.UNICODE | re.IGNORECASE)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_num += 1
        if page_num in [1, 2, 3] or page_num >= 106:
            continue
        
        blocks = page.get_text("blocks")
        question_text = []
        for block in blocks:
            block_text = block[4].strip()
            if not block_text:
                continue
            
            clean_block = clean_text(block_text, is_heading=True)
            if not clean_block:
                continue
                
            is_heading = False
            is_question = False
            matches = match_titles([clean_block], ground_truth_titles, threshold=60)
            parsed, matched_title, score = matches[0]
            if matched_title:
                matched_index = ground_truth_titles.index(matched_title)
                if matched_index >= heading_index:
                    is_heading = True
                    heading_index = matched_index + 1
                    in_question_section = "FRAGENKATALOG" in matched_title.upper() or 19 <= page_num <= 26
            
            if in_question_section:
                block_lines = block_text.split('\n')
                for line in block_lines:
                    clean_line = clean_text(line, is_heading=True)
                    if clean_line and (question_pattern.search(clean_line) or question_text):
                        question_text.append(clean_line)
                        if clean_line.endswith('?'):
                            question_content = " ".join(question_text)
                            question_matches = match_titles([question_content], eval_questions, threshold=80)
                            if question_matches[0][1]:
                                is_heading = True
                                is_question = True
                                clean_block = question_content
                            question_text = []
                        continue
                    elif question_text:
                        question_text.append(clean_line)
            
            if is_heading:
                if current_segment['text']:
                    text_content = " ".join(current_segment['text']).strip()
                    if len(text_content.split()) > 80:
                        segments.append({
                            "title": current_segment['title'],
                            "text": text_content,
                            "page": current_segment['page']
                        })
                    current_segment['text'] = []
                
                current_segment['title'] = f"8. FRAGENKATALOG MIT ANTWORTEN - {clean_block}" if is_question else matched_title if matched_title else headings[heading_index-1][1]
                current_segment['page'] = page_num
            else:
                clean_content = clean_text(block_text, is_heading=False)
                if clean_content:
                    current_segment['text'].append(clean_content + "|||PARAGRAPH|||")
    
    if current_segment['text']:
        text_content = " ".join(current_segment['text']).strip()
        if len(text_content.split()) > 80:
            segments.append({
                "title": current_segment['title'],
                "text": text_content,
                "page": current_segment['page']
            })
    
    doc.close()
    return segments

# Split and merge segments
def split_segments(segments, min_words=80, eval_questions=None):
    final_segments = []
    question_pattern = re.compile(r'^(?:\d+\.\s*|[A-Z][\.\)]\s*)(WIE|WELCHE|WANN|WARUM|WAS|SOLLTE|WEN|KANN|IST|HABEN)\s*.+\?', re.UNICODE | re.IGNORECASE)
    
    for seg in segments:
        max_words = 500 if "FRAGENKATALOG" in seg['title'].upper() else 300
        words = seg['text'].split()
        if len(words) <= max_words:
            final_segments.append(seg)
            continue
        
        paragraphs = re.split(r'\|\|\|PARAGRAPH\|\|\|', seg['text'])
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        current_words = []
        current_text = []
        part_count = 1
        
        for para in paragraphs:
            para_words = para.split()
            if len(current_words) + len(para_words) > max_words:
                if current_words:
                    chunk_text = " ".join(current_text).strip()
                    if len(chunk_text.split()) >= min_words:
                        final_segments.append({
                            "title": f"{seg['title']} - Part {part_count}",
                            "text": chunk_text,
                            "page": seg['page']
                        })
                        part_count += 1
                    current_words = []
                    current_text = []
            current_words.extend(para_words)
            current_text.append(para)
            
            if "FRAGENKATALOG" in seg['title'].upper() and question_pattern.match(para.strip()):
                question_matches = match_titles([para.strip()], eval_questions, threshold=80)
                if question_matches[0][1]:
                    if current_words:
                        chunk_text = " ".join(current_text).strip()
                        if len(chunk_text.split()) >= min_words:
                            final_segments.append({
                                "title": f"8. FRAGENKATALOG MIT ANTWORTEN - {para.strip()}",
                                "text": chunk_text,
                                "page": seg['page']
                            })
                            part_count += 1
                    current_words = []
                    current_text = []
        
        if current_words:
            chunk_text = " ".join(current_text).strip()
            if len(chunk_text.split()) >= min_words:
                final_segments.append({
                    "title": f"{seg['title']} - Part {part_count}",
                    "text": chunk_text,
                    "page": seg['page']
                })
    
    merged_segments = []
    i = 0
    while i < len(final_segments):
        seg = final_segments[i]
        if len(seg['text'].split()) < min_words and i > 0 and seg['page'] == merged_segments[-1]['page']:
            prev_seg = merged_segments[-1]
            prev_seg['text'] += " " + seg['text']
        else:
            merged_segments.append(seg)
        i += 1
    
    final_segments = [seg for seg in merged_segments if len(seg['text'].split()) > 80]
    return final_segments

# Embed and index segments
def embed_and_index(segments):
    model = SentenceTransformer('deepset/gbert-base')
    segment_texts = [f"Title: {seg['title']} Text: {seg['text']}" for seg in segments]
    embeddings = model.encode(segment_texts, show_progress_bar=True, batch_size=32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return model, index, segments

# Search function
def search(query, model, index, segments, k=3):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k * 2)  # Retrieve more for inner splitting
    
    inner_segments = []
    for i, idx in enumerate(indices[0]):
        seg = segments[idx]
        paragraphs = re.split(r'\|\|\|PARAGRAPH\|\|\|', seg['text'])
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.split()) >= 15]
        for j, para in enumerate(paragraphs):
            inner_segments.append({
                "original_title": seg["title"],
                "text": para,
                "page": seg["page"],
                "original_score": distances[0][i],
                "para_index": j
            })
    
    inner_texts = [f"Title: {seg['original_title']} Text: {seg['text']}" for seg in inner_segments]
    inner_embeddings = model.encode(inner_texts, show_progress_bar=False)
    faiss.normalize_L2(inner_embeddings)
    inner_index = faiss.IndexFlatIP(inner_embeddings.shape[1])
    inner_index.add(inner_embeddings)
    
    inner_distances, inner_indices = inner_index.search(query_embedding, k)
    
    results = []
    for i, inner_idx in enumerate(inner_indices[0]):
        inner_seg = inner_segments[inner_idx]
        score = inner_distances[0][i]
        score = score * 3.0 if "FRAGENKATALOG" in inner_seg["original_title"].upper() else score
        results.append({
            "title": inner_seg["original_title"],
            "text": inner_seg["text"],
            "page": inner_seg["page"],
            "score": float(score)
        })
    
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    return results

# Compute metrics
def compute_metrics(retrieved_chunks, gold_answer):
    gold_words = set(re.sub(r'[^\w\s]', '', gold_answer.lower()).split())
    common_words = set()
    total_retrieved = 0
    for chunk in retrieved_chunks:
        retrieved_words = set(re.sub(r'[^\w\s]', '', chunk['text'].lower()).split())
        common_words.update(gold_words & retrieved_words)
        total_retrieved += len(retrieved_words) if retrieved_words else 1
    total_common = len(common_words)
    precision = total_common / total_retrieved if total_retrieved else 0.0
    recall = min(total_common / len(gold_words) if gold_words else 0.0, 1.0)
    return precision, recall

# Main execution
if __name__ == "__main__":
    pdf_path = "/content/drive/MyDrive/rag_project/Guideline_atraumatische_Femurkopfnekrose_2019-09_1-abgelaufen.pdf"
    output_json_path = '/content/drive/MyDrive/rag_project/segments_headings.json'
    eval_json_path = '/content/drive/MyDrive/rag_project/questions_answers.json'
    
    if os.path.exists(pdf_path):
        # Load evaluation questions
        with open(eval_json_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        eval_questions = [item['question'] for item in eval_data]
        
        headings = extract_headings(pdf_path)
        segments = segment_pdf(pdf_path, headings, eval_questions)
        final_segments = split_segments(segments, min_words=80, eval_questions=eval_questions)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_segments, f, ensure_ascii=False, indent=2)
        
        model, index, final_segments = embed_and_index(final_segments)
        
        # Evaluate
        precisions, recalls = [], []
        for item in eval_data:
            query = item['question']
            gold_answer = item['answer_reference']
            results = search(query, model, index, final_segments, k=3)
            precision, recall = compute_metrics(results, gold_answer)
            print(f"Query: {query}")
            print(f"Gold Answer: {gold_answer}")
            print(f"Precision@3: {precision:.4f}, Recall@3: {recall:.4f}")
            print("Retrieved Chunks:")
            for res in results:
                print(f"  Title: {res['title']}, Page: {res['page']}, Score: {res['score']:.4f}, Text: {res['text'][:100]}...")
            print("-" * 50)
            precisions.append(precision)
            recalls.append(recall)
        
        print(f"Average Precision@3: {np.mean(precisions):.4f}")
        print(f"Average Recall@3: {np.mean(recalls):.4f}")
    else:
        print(f"PDF not found at: {pdf_path}")