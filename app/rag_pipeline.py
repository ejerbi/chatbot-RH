from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import BartForConditionalGeneration, BartTokenizer
import atexit
import torch
torch.set_num_threads(1)

# Load Documents (Étape 1)
def load_documents(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        documents.append(text)
    return documents

# Split Documents (Étape 2)
def split_documents(documents):
    chunks = []
    for document in documents:
        # Séparer le texte en questions-réponses
        questions_reponses = document.split("\nQ : ")
        
        for qr in questions_reponses:
            qr = qr.strip()
            if qr:  # Éviter les entrées vides
                chunks.append("Q : " + qr) 
    return chunks

# Store Embeddings (Étape 3)
def store_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Utilisation de Hugging Face
    embeddings = model.encode(chunks, convert_to_tensor=True)

    # Conversion des embeddings en format numpy pour FAISS
    embeddings_np = embeddings.cpu().numpy()
    faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Index pour la recherche
    faiss_index.add(embeddings_np)  # Ajout des embeddings dans l'index

    return faiss_index, embeddings_np

# Retrieve Relevant Documents (Étape 4)
def retrieve_relevant_documents(query, faiss_index, chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Charger le modèle d'embedding
    query_embedding = model.encode([query])  # Générer l'embedding de la requête

    # Recherche dans FAISS
    results = faiss_index.search(query_embedding, k=1)  # k=1 pour récupérer le meilleur résultat

    # Récupération des chunks pertinents
    relevant_chunks = [chunks[i] for i in results[1].flatten()]  

    # Sélection du chunk le plus pertinent
    if relevant_chunks:
        best_chunk = relevant_chunks[0]  # Prendre uniquement le premier
    else:
        best_chunk = "Je ne trouve pas de réponse pertinente."

    print(f"Chunk sélectionné : {best_chunk}")
    # print("Chunks récupérés :", relevant_chunks)  # À décommenter pour debug

    return relevant_chunks

# Generate Response (Étape 5) - Remplacer OpenAI par BART de Hugging Face
def generate_response(relevant_chunks):
    # Charger le modèle BART et le tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Combiner les chunks pertinents pour le prompt
    prompt = " ".join(relevant_chunks)
    
    # Tokenisation du prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    # Générer la réponse avec BART
    summary_ids = model.generate(inputs['input_ids'], max_length=200, num_beams=4, no_repeat_ngram_size=2, temperature=0.7)
    
    # Décoder la réponse générée
    response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return response

# Exemple d'utilisation
pdf_path = "data/FAQ_RH.pdf"
documents = load_documents(pdf_path)
chunks = split_documents(documents)
faiss_index, embeddings = store_embeddings(chunks)

# Ensure proper cleanup of FAISS resources

atexit.register(faiss_index.reset)

query = "Quel est le processus pour un entretien de recrutement ?"
try:
    relevant_chunks = retrieve_relevant_documents(query, faiss_index, chunks)
    response = generate_response(relevant_chunks)
    print(response)
finally:
    # Explicitly clean up FAISS resources
    del faiss_index