import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import re
import os

torch.set_num_threads(1)

# ---- 1. Extraction et nettoyage du texte du PDF ----
def load_documents(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces inutiles
            documents.append(text)
    return documents

# ---- 2. Séparation des questions-réponses ----
def split_documents(documents):
    chunks = []
    for document in documents:
        questions_reponses = re.split(r"\bQ\s*:\s*", document)
        for qr in questions_reponses:
            qr = qr.strip()
            if "R :" in qr:
                qr = "Q : " + qr  
                chunks.append(qr)
    return chunks

# ---- 3. Stockage des embeddings avec FAISS ----
def store_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)

    embeddings_np = embeddings.cpu().numpy()
    faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])
    faiss_index.add(embeddings_np)

    return faiss_index, embeddings_np, model

# ---- 4. Recherche des documents pertinents ----
def retrieve_relevant_documents(query, faiss_index, chunks, embedding_model):
    query_embedding = embedding_model.encode([query])

    results = faiss_index.search(query_embedding, k=1)  # Récupérer les 3 meilleurs résultats
    relevant_chunks = [chunks[i] for i in results[1].flatten()]

    return relevant_chunks if relevant_chunks else ["Désolé, je n'ai pas trouvé de réponse pertinente."]

# ---- 5. Génération de la réponse avec BART ----
def generate_response(relevant_chunks, bart_model, bart_tokenizer):
    context = []
    for chunk in relevant_chunks:
        if "Q :" in chunk:
            # Supprimer tous les caractères jusqu'au prochain "."
            chunk = re.sub(r'Q :.*?\?', '', chunk, count=1).strip()
        if "R :" in chunk:
            # Garder uniquement les parties pertinentes
            chunk = chunk.replace("R :", "").strip()
        context.append(chunk)

    context = " ".join(context)
    context = re.sub(r'\s+', ' ', context).strip()

    inputs = bart_tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    print(hasattr(bart_tokenizer, "lang_code_to_id"))
    
    summary_ids = bart_model.generate(
        inputs['input_ids'], 
        max_length=150, 
        num_beams=4, 
        no_repeat_ngram_size=2, 
        do_sample=True, 
        temperature=0.2,
        forced_bos_token_id=bart_tokenizer.convert_tokens_to_ids("<s>")
            # Forcer la génération en français
    )
    
    response = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return response

# ---- 6. Chargement du modèle BART une seule fois ----
model_name = "facebook/bart-large-cnn"
bart_model = BartForConditionalGeneration.from_pretrained(model_name)
bart_tokenizer = BartTokenizer.from_pretrained(model_name)

# ---- 7. Interface Streamlit ----
st.title("Chatbot FAQ - RH")

# ---- Champ pour entrer le chemin du fichier PDF ----
pdf_path = "data/FAQ_RH.pdf"  # Chemin par défaut
if pdf_path:
    if os.path.exists(pdf_path):
        st.spinner("Lecture du fichier...")
        documents = load_documents(pdf_path)
        chunks = split_documents(documents)
        faiss_index, embeddings, embedding_model = store_embeddings(chunks)
        st.session_state["faiss_index"] = faiss_index
        st.session_state["chunks"] = chunks
        st.session_state["embedding_model"] = embedding_model
        st.success("Fichier chargé avec succès ! Posez votre question.")
    else:
        st.error("Chemin invalide ! Veuillez vérifier l'emplacement du fichier.")

# ---- Gestion de l'historique des messages ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Affichage de l'historique ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Zone de saisie utilisateur ----
query = st.chat_input("Posez une question ici...")

if query and "faiss_index" in st.session_state:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # ---- Récupération et génération de la réponse ----
    with st.spinner("Recherche de la réponse..."):
        relevant_chunks = retrieve_relevant_documents(query, 
                                                      st.session_state["faiss_index"], 
                                                      st.session_state["chunks"], 
                                                      st.session_state["embedding_model"])
        response = generate_response(relevant_chunks, bart_model, bart_tokenizer)

    # ---- Affichage de la réponse ----
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})