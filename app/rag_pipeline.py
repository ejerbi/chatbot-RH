from PyPDF2 import PdfReader
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import re

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
        # Séparer avec un regex plus robuste
        questions_reponses = re.split(r"\bQ\s*:\s*", document)
        
        for qr in questions_reponses:
            qr = qr.strip()
            if "R :" in qr:
                qr = "Q : " + qr  # Reformater la question-réponse
                chunks.append(qr)
    return chunks

# ---- 3. Stockage des embeddings avec FAISS ----
def store_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)

    # Stockage dans FAISS
    embeddings_np = embeddings.cpu().numpy()
    faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])
    faiss_index.add(embeddings_np)

    return faiss_index, embeddings_np

# ---- 4. Recherche des documents pertinents ----
def retrieve_relevant_documents(query, faiss_index, chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    results = faiss_index.search(query_embedding, k=3)  # Récupérer les 3 meilleurs résultats
    relevant_chunks = [chunks[i] for i in results[1].flatten()]

    if relevant_chunks:
        print(f"Documents pertinents trouvés : {relevant_chunks}")
        return relevant_chunks
    else:
        return ["Désolé, je n'ai pas trouvé de réponse pertinente."]

# ---- 5. Génération de la réponse avec BART ----
def generate_response(relevant_chunks):
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Nettoyer les chunks avant de les envoyer à BART
    context = " ".join(relevant_chunks)
    context = re.sub(r'\s+', ' ', context).strip()

    # Encodage
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    # Génération
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=150, 
        num_beams=4, 
        no_repeat_ngram_size=2, 
        do_sample=True, 
        temperature=0.7
    )
    
    response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return response

# ---- 6. Exemple d'utilisation ----
pdf_path = "FAQ_RH.pdf"
documents = load_documents(pdf_path)
chunks = split_documents(documents)
faiss_index, embeddings = store_embeddings(chunks)

query = "Puis-je reporter mes conges a l'annee suivante ?"

try:
    relevant_chunks = retrieve_relevant_documents(query, faiss_index, chunks)
    response = generate_response(relevant_chunks)
    print(response)
finally:
    del faiss_index  # Libération de la mémoire

