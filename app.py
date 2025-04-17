import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
from chromadb.config import Settings
import torch

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chatbot – INSIEL")

@st.cache_resource
def load_models():
    # Embedding model
    embedder = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

    # LLM model (TinyLlama su CPU)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(torch.device("cpu"))
    rag_chat = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300, device=-1)

    return embedder, rag_chat

embedder, rag_chat = load_models()

# --- CHROMA DB SETUP ---
#client = chromadb.PersistentClient(path="./vectorstore")

#collection = client.get_or_create_collection("insiel_chunks")

client = chromadb.PersistentClient(path="./vectorstore")
collection = client.get_or_create_collection(name="insiel_chunks")

# --- FUNZIONE DI RISPOSTA ---

def generate_rag_response_local(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    context = context[:3000]  # taglia se troppo lungo per evitare overflow

    prompt = (
        "Rispondi alla domanda usando solo le informazioni nel contesto. "
        "Se la risposta non è presente, di' chiaramente che non è specificato nel documento.\n\n"
        f"Contesto:\n{context}\n\n"
        f"Domanda: \n{query}\n"
        "Risposta:"
    )

    result = rag_chat(prompt)[0]["generated_text"]

    return result.split("Risposta:")[-1].strip()


# --- INTERFACCIA ---

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("💬 Inserisci la tua domanda qui:")

if query:
    # 1. Embedding della query
    query_embedding = embedder.encode([query])

    # 2. Retrieval da Chroma
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    retrieved_chunks = results["documents"][0]

    # 3. Risposta con modello locale
    response = generate_rag_response_local(query, retrieved_chunks)

    # 4. Aggiorna cronologia chat
    st.session_state.history.append(("🧑‍💻 Tu", query))
    response_preview = "\n".join(response.strip().split("\n")[:2])
    st.session_state.history.append(("🤖 RAG Bot", response_preview))

# --- OUTPUT CHAT ---

if st.session_state.history:
    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}**: {msg}")

# --- VISUALIZZA I CHUNK USATI ---
if query:
    with st.expander("🔍 Mostra i documenti/chunk usati"):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}**\n\n{chunk}\n\n---")
