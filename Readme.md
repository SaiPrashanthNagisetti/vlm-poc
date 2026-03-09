# Multimodal RAG Pipeline for Document 

## Overview

This project implements a **Multimodal Retrieval-Augmented Generation (RAG) pipeline** for understanding complex PDF documents that may contain:

* Text
* Tables
* Images
* Vector graphics
* Charts

The system extracts structured information from documents, analyzes visual elements using a Vision Language Model (VLM), merges multimodal information, and enables question answering using a vector database and large language models.

The pipeline is designed to work with documents where information may appear across **text, tables, and visual elements simultaneously**.

---

# System Architecture

```
PDF Documents
      │
      ▼
Metadata Extraction
      │
      ▼
PDF Asset Extraction
(Text, Tables, Images, Vector Graphics)
      │
      ▼
Vision-Language Image Analysis
      │
      ▼
Multimodal JSON Merge
(Text + Visual Summaries)
      │
      ▼
Vector Index (ChromaDB)
      │
      ▼
RAG Query Engine
      │
      ▼
Answer Generation with Citations
```

---

# Project Structure

```
VLM_POC/
│
├── config/
│   └── settings.py
│
├── processors/
│   ├── image_processor.py
│   ├── pdf_content_detector.py
│   ├── pdf_extractor.py
│   └── pdf_metadata_processor.py
│
├── services/
│   ├── image_analyzer.py
│   └── openai_client.py
│
├── pipelines/
│   ├── merged_json.py
│   └── rag_pipeline.py
│
├── utils/
│   ├── encoder.py
│   └── file_utils.py
│
├── input/
│   └── PDF documents
│
├── extracted_assets/
│   ├── tables/
│   ├── images/
│   ├── vector_graphics/
│   └── text/
│
├── pdf_metadata_output/
│   └── metadata JSON files
│
├── output/
│   └── image analysis results (JSON)
│
├── merged_output/
│   └── merged multimodal JSON files
│
├── vector_store/
│   └── ChromaDB persistent storage
│
├── evaluation_results/
│   └── evaluation outputs
│
├── evaluate.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# Pipeline Components

## 1 Metadata Extraction

File:

```
processors/pdf_metadata_processor.py
```

Extracts document-level metadata before the main extraction pipeline.

Output:

```
pdf_metadata_output/
    document_metadata.json
```

---

# 2 PDF Asset Extraction

File:

```
processors/pdf_extractor.py
```

Extracts structured elements from PDFs including:

* Page text
* Tables
* Images
* Vector graphics

Libraries used:

* **pdfplumber**
* **PyMuPDF**

Output stored in:

```
extracted_assets/
```

Example:

```
extracted_assets/
    document_name/
        tables/
        images/
        vector_graphics/
        text/
```

---

# 3 Image Processing

File:

```
processors/image_processor.py
```

Coordinates analysis of extracted visual assets.

Uses:

```
services/image_analyzer.py
```

to analyze images using a Vision Language Model.

The image analysis produces:

* image titles
* keywords
* summaries
* structured table rows

Results are stored in:

```
output/
```

Example output:

```json
{
  "file_name": "document_name",
  "page_number": 1,
  "image_title": "...",
  "image_keywords": [...],
  "summary": "...",
  "table_rows": [...]
}
```

---

# 4 Multimodal JSON Merge

File:

```
pipelines/merged_json.py
```

Combines:

* extracted text
* visual summaries
* metadata
* keywords

into structured page-level JSON documents.

Output stored in:

```
merged_output/
```

Example:

```json
{
  "pdf": "document_name",
  "page": 1,
  "text": "...",
  "keywords": [...],
  "visual_summaries": [...]
}
```

---

# 5 Vector Database Indexing

File:

```
pipelines/rag_pipeline.py
```

Documents are indexed using:

* **OpenAI Embeddings**
* **LlamaIndex**
* **ChromaDB**

Vector index stored in:

```
vector_store/
```

---

# 6 Retrieval-Augmented Generation

User queries are processed using:

1. Query embedding
2. Vector retrieval
3. Context-aware prompt
4. LLM reasoning

The system returns:

* generated answer
* supporting context
* source citations
* related visual references

Example output:

```json
{
  "answer": "...",
  "sources": [
    {
      "pdf": "document_name",
      "page": 1
    }
  ]
}
```

---

# Evaluation

The repository includes an evaluation pipeline.

Dataset format:

```json
{
  "question": "...",
  "answer": "..."
}
```

Run evaluation:

```
python evaluate.py
```

Results are saved to:

```
evaluation_results/
```

Example output:

```json
{
  "question": "...",
  "expected_answer": "...",
  "predicted_answer": "...",
  "status": "PASS"
}
```

---

# Installation

## 1 Clone the repository

```
git clone https://github.com/your-repository-name.git
cd VLM_POC
```

---

## 2 Create a virtual environment

```
python -m venv .venv
```

Activate it.

Windows:

```
.venv\Scripts\activate
```

Mac/Linux:

```
source .venv/bin/activate
```

---

## 3 Install dependencies

```
pip install -r requirements.txt
```

---

## 4 Configure environment variables

Create a `.env` file in the project root.

```
OPENAI_API_KEY=your_api_key
```

---

# Running the Pipeline

### Step 1 — Add documents

Place PDF documents in:

```
input/
```

---

### Step 2 — Extract metadata

```
python processors/pdf_metadata_processor.py
```

---

### Step 3 — Extract PDF assets

```
python processors/pdf_extractor.py
```

---

### Step 4 — Run image analysis

```
python processors/image_processor.py
```

---

### Step 5 — Merge multimodal data

```
python pipelines/merged_json.py
```

---

### Step 6 — Build vector index

```
python pipelines/rag_pipeline.py
```

---

### Step 7 — Run queries

Use the RAG query pipeline to ask questions about the documents.

---

# Technologies Used

* Python
* PyMuPDF
* pdfplumber
* scikit-learn
* LlamaIndex
* ChromaDB
* OpenAI API
* Vision Language Models
* TF-IDF keyword extraction

---

# Future Improvements

Potential enhancements include:

* Layout-aware chunking
* Hybrid retrieval (keyword + vector search)
* Table-aware embeddings
* Improved chart interpretation
* Local LLM integration with Ollama
* Advanced RAG evaluation metrics
