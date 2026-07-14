import os
import time
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── GROQ CONFIG ────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── STEP 1: LOAD ALL 5 PDFs ────────────────────────────────
print("Loading PDFs...")
documents = []
for filename in os.listdir("data"):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{filename}")
        documents.extend(loader.load())
print(f"Total pages loaded: {len(documents)}")

# ── STEP 2A: CHUNKING STRATEGY A (Small - 500 chars) ───────
splitter_A = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks_A = splitter_A.split_documents(documents)
print(f"\nStrategy A - Small chunks: {len(chunks_A)}")

# ── STEP 2B: CHUNKING STRATEGY B (Large - 1500 chars) ──────
splitter_B = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks_B = splitter_B.split_documents(documents)
print(f"Strategy B - Large chunks: {len(chunks_B)}")

# ── STEP 3: EMBEDDINGS ─────────────────────────────────────
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embeddings ready.")

# ── STEP 4: BUILD 2 VECTOR DATABASES ──────────────────────
print("\nBuilding database for Strategy A...")
db_A = FAISS.from_documents(chunks_A, embeddings)

print("Building database for Strategy B...")
db_B = FAISS.from_documents(chunks_B, embeddings)

print("Both databases ready!")

# ── STEP 5: 20 QUESTIONS ──────────────────────────────────
questions = [
    "What is the main objective of the Farmer Connect platform?",
    "What algorithm does Farmer Connect use to allocate produce fairly among farmers?",
    "What are the four user roles supported by the Farmer Connect platform?",
    "How does Farmer Connect support farmers who do not own smartphones?",
    "What is the main problem FlowEdit tries to solve in text-to-speech systems?",
    "What type of memory does FlowEdit use to store pronunciation corrections?",
    "By what percentage does FlowEdit reduce Phoneme Error Rate compared to the zero-shot baseline?",
    "How long does it take FlowEdit to complete one pronunciation correction on a single GPU?",
    "What are the three hypotheses tested about how compliance demonstrations affect LLM behavior?",
    "Which training stage was found to prevent benign demonstrations from increasing harmful compliance?",
    "Which model showed the strongest robustness against many-shot demonstrations?",
    "What ordering of demonstrations leads to the highest compliance rate in models?",
    "What framework does the cross-attention TTS paper adapt to the speech domain for the first time?",
    "Which token category shows the lowest temporal variance in cross-attention analysis?",
    "At which transformer layer does style-token attention importance peak?",
    "Which style word shows the strongest correlation with energy in the acoustic analysis?",
    "What are the three main modules in the Lagrange architecture?",
    "What collision rate does Lagrange achieve on the CODA benchmark for out-of-distribution scenarios?",
    "How does Lagrange handle sensor failures like a camera dropout?",
    "What limitation does Lagrange have regarding non-geometric hazards?"
]

for i, query in enumerate(questions):
    print(f"\n{'='*60}")
    print(f"QUESTION {i+1}: {query}")
    print('='*60)

    # CONFIG 1: Small Chunks + Cosine Similarity
    results1 = db_A.similarity_search(query, k=3)
    context1 = "\n\n".join([doc.page_content for doc in results1])
    print("\nCONFIG 1: Small Chunks + Cosine Similarity")
    print("-"*40)
    time.sleep(2)
    print(ask_llm(f"Answer only from context.\n\nContext:\n{context1}\n\nQuestion: {query}\nAnswer:"))

    # CONFIG 2: Small Chunks + MMR
    results2 = db_A.max_marginal_relevance_search(query, k=3, fetch_k=10)
    context2 = "\n\n".join([doc.page_content for doc in results2])
    print("\nCONFIG 2: Small Chunks + MMR")
    print("-"*40)
    time.sleep(2)
    print(ask_llm(f"Answer only from context.\n\nContext:\n{context2}\n\nQuestion: {query}\nAnswer:"))

    # CONFIG 3: Large Chunks + Cosine Similarity
    results3 = db_B.similarity_search(query, k=3)
    context3 = "\n\n".join([doc.page_content for doc in results3])
    print("\nCONFIG 3: Large Chunks + Cosine Similarity")
    print("-"*40)
    time.sleep(2)
    print(ask_llm(f"Answer only from context.\n\nContext:\n{context3}\n\nQuestion: {query}\nAnswer:"))

    # CONFIG 4: Large Chunks + MMR
    results4 = db_B.max_marginal_relevance_search(query, k=3, fetch_k=10)
    context4 = "\n\n".join([doc.page_content for doc in results4])
    print("\nCONFIG 4: Large Chunks + MMR (Advanced)")
    print("-"*40)
    time.sleep(2)
    print(ask_llm(f"Answer only from context.\n\nContext:\n{context4}\n\nQuestion: {query}\nAnswer:"))

    # WITHOUT RAG
    print("\nWITHOUT RAG (LLM alone)")
    print("-"*40)
    time.sleep(2)
    print(ask_llm(query))

print("\n" + "="*60)
print("ALL 20 QUESTIONS COMPLETE!")
print("="*60)