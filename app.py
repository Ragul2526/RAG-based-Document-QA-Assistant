import fitz  # PyMuPDF
import numpy as np
import faiss
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading tokenizer and LLM (this can take a minute on first run)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    device_map="auto" if device == "cuda" else None
)
if device == "cpu":
    model.to(device)
model.eval()
print(f"Models loaded. Running on: {device}")

current_chunks = []
current_index = None

def chunk_text(pages, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
  chunks = []
  chunk_index = 0
  for page_num, page_text in enumerate(pages):
    page_text = page_text.strip()
    if not page_text:
      continue
    s = 0
    while s < len(page_text):
      e = s + chunk_size
      chunk = page_text[s:e]

      chunks.append(
          {
           "text" : chunk,
           "page_number" : page_num + 1,
           "chunk_index" : chunk_index
          }
      )
      chunk_index += 1
      s += chunk_size - chunk_overlap
  return chunks
def process_pdf(pdf_file):
    
    """
    The function extracts  text, chunk, embed, and build a fresh FAISS index for the uploaded PDF
    """
    global current_chunks, current_index
 
    if pdf_file is None:
        return "Please upload a PDF first."
    doc = fitz.open(pdf_file.name)
    pages = [page.get_text() for page in doc]
    total_pages = len(pages)

    all_text = " ".join(pages).strip()
    if not all_text:
       return "This PDF appears to be scanned/image-based. No extractable text found."
    current_chunks = chunk_text(pages)
    chunk_texts = [c["text"] for c in current_chunks]
 
    embeddings = embedder.encode(chunk_texts, show_progress_bar=False)
    embeddings_f32 = np.array(embeddings).astype("float32")
 
    dimension = embeddings_f32.shape[1]
    current_index = faiss.IndexFlatL2(dimension)
    current_index.add(embeddings_f32)
 
    return (
        f"   PDF processed successfully!\n"
        f"   Pages: {total_pages}\n"
        f"   Chunks created: {len(current_chunks)}\n"
        f"   Ready to answer questions!"
    )

def retrieve(question, k = 3):
    """
      When giving a plain question the function returns the top k most relevent chunks.
    
      Input :
        question : the user's question
        k : how many chunks to return
    
      Output :
        list of dicts, each with keys: text, page_number, chunk_index, distance
    
    """
    query_vector = embedder.encode([question]).astype("float32")
    distances, indices = current_index.search(query_vector, k)
 
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = current_chunks[idx].copy()
        chunk["distance"] = round(float(dist), 4)
        results.append(chunk)
    return results
def build_rag_prompt(question,  retrieved_chunks):
    """
      This function combines retrieved chunks + question into a chat-style prompt.
    
      Input:
        question - User's question
        retrieved_chunks - Output of retrieve()
    
      Output:
        list of message dicts in OpenAI/HuggingFace chat format
    
    """
    context_blocks = []
    for i,chunk in enumerate(retrieved_chunks):
        block = f"[Source {i+1} - Page {chunk['page_number']}]\n{chunk['text'].strip()}"
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)
    #print(f"\nContext:{context_str}\n")
    messages = [
        {
            "role": "user",
            "content": (
                f"Here is context extracted from a document:\n\n"
                f"{context_str}\n\n"
                f"---\n"
                f"Using ONLY the context above, answer this question. "
                f"Do NOT use outside knowledge. "
                f"If the answer is not in the context, say 'I could not find that in the document.'\n\n"
                f"Question: {question}"
                )
            }
        ]


    return messages

def gen_answer(question, k = 3, max_new_tokens = 300):
    """
      This Function represents the full RAG pipeline: retrieve ->  prompt -> generate -> return answer to user
    
      Input:
        question - user's question
        k - how many chunks to retrieve
        max_new_tokens : max length of the answer
    
      Output:
        dict with keys: answer, retrieved_chunks
    
    """
    retrieved = retrieve(question, k)
    messages = build_rag_prompt(question, retrieved)

    tokenized = tokenizer.apply_chat_template( # apply_chat_template formats messages into the exact string Qwen expects
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(device)
    if isinstance(tokenized, dict) or hasattr(tokenized, 'input_ids'):
        input_ids = tokenized["input_ids"].to(device)
    else:
        input_ids = tokenized.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding - deterministic output
            temperature=None,       # must be None when do_sample=False
            top_p=None,             # must be None when do_sample=False
            pad_token_id=tokenizer.eos_token_id)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    sources = ""
    for i, chunk in enumerate(retrieved):
        preview = chunk["text"][:150].replace("\n", " ")
        sources += (
            f"Source {i+1} - Page {chunk['page_number']} "
            f"(dist={chunk['distance']})\n{preview}...\n\n"
            )
 
    return answer, sources

with gr.Blocks(title = "RAG Document QA", theme = gr.theme.Soft()) as demo:
    gr.Markdown("""
    #RAG-based Document QA Assistant
    Upload any PDF → Ask questions → Get answers grounded in the document, with sources.
 
    *Running on CPU — answers may take 30–90 seconds.*
                """)
    with gr.Row():
        with gr.Column(scale = 1):
            pdf_input = gr.File(label= "Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant= "primary")
            process_stat = gr.Textbox(label="Status", lines = 4, interactive= False )
        with gr.Column(scale = 2):
            question = gr.Textbox(label="Ask a question",
                                  placeholder="for example: what is the candidates location?",
                                  lines=2)
            num_chunks = gr.Slider(
                minimum=1,maximum= 6,step =1,
                label="Number of chunks to retrieve(k)")
            ask_btn = gr.Button("Get Answer", varaint = "primary")
            answer_output = gr.Textbox(label = "Answer",line = 6, interactive = False)
            source_output = gr.Textbox(label = "Retrieved source", line = 8, interactive = False)
 
    process_btn.click(fn = process_pdf, inputs = [pdf_input], outputs = [process_stat])
    ask_btn.click(fn = gen_answer, inputs = [question, num_chunks], outputs = [answer_output, source_output]  )
    question.submit(fn=gen_answer, inputs=[question, num_chunks], outputs=[answer_output, source_output])
           
if __name__ == "__main__":
    demo.launch()          