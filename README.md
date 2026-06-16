# RAG-based Document QA Assistant

A simple Retrieval-Augmented Generation (RAG) pipeline that answers questions about a PDF document by retrieving relevant chunks and passing them as context to an LLM.

**🚀 Live demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/roy2526/rag-document-qa)

![Demo screenshot](<img width="1526" height="880" alt="hf_output" src="https://github.com/user-attachments/assets/6eb4119e-651d-45d8-9d24-c5a1caeab6ce" />
)

## What this is

Upload any PDF, ask a question about it, and get an answer grounded in the actual document — with the exact source chunks and page numbers shown alongside the answer. No fine-tuning, no external API calls, runs entirely on free-tier infrastructure.

## How it works

```
PDF upload
   ↓
Extract text (PyMuPDF)
   ↓
Chunk with overlap
   ↓
Embed with all-MiniLM-L6-v2
   ↓
Store in FAISS index
   ↓
User asks question
   ↓
Embed question → search FAISS → top-K chunks
   ↓
Build grounded prompt
   ↓
Qwen2.5-1.5B-Instruct generates answer
   ↓
Display answer + sources
```

## Repo structure

| Path | What it is |
|---|---|
| `notebook/RAG_Document_QA.ipynb` | Step-by-step Colab notebook - each section pairs a markdown explanation with runnable code, built for learning the pipeline piece by piece |
| `app.py` | Standalone Gradio app deployed to Hugging Face Spaces |
| `requirements.txt` | Dependencies for the Space |

## Tech stack

- **PDF parsing:** PyMuPDF
- **Embeddings:** [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers)
- **Vector search:** FAISS (`IndexFlatL2`)
- **Generation:** [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **UI:** Gradio
- **Hosting:** Hugging Face Spaces (free CPU tier)

## Why RAG, not just an LLM

A 1.5B model has never seen your private PDF - ask it directly and it correctly says it doesn't know. RAG changes that: by retrieving the actual relevant text and handing it to the model as context, a model that has never seen your document becomes able to answer questions about it accurately, with citations back to the source pages. That's the core value RAG provides - not just reducing hallucination, but making private-document Q&A possible in the first place.

## Running it yourself

### Option 1 — Colab notebook
Open `notebook/RAG_Document_QA.ipynb` in Google Colab, set the runtime to a GPU (Runtime -> Change runtime type -> T4 GPU), and run cells top to bottom. Upload your own PDF when prompted in Step 1.

### Option 2 — Local / Hugging Face Space
```bash
pip install -r requirements.txt
python app.py
```
This launches the same Gradio app used in the live demo above. Runs on CPU; expect 30–90 seconds per answer without a GPU.

## Notes on free-tier deployment

This Space runs on Hugging Face's free CPU hardware, so:
- Answers take roughly 30–90 seconds (no GPU)
- The Space sleeps after inactivity and takes ~30s to wake on the next visit
- No paid API calls are used - the model runs locally inside the Space

## Possible next steps

- Swap `IndexFlatL2` for `IndexIVFFlat` to scale to much larger documents
- Add support for multiple PDFs / a persistent document library
- Stream tokens as they're generated instead of waiting for the full answer
