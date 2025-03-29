from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI  # Removed as we are using Hugging Face pipeline
from transformers import pipeline

# Load Documents (Étape 1)
def load_documents(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        documents.append(text)
    return documents

# Split Documents (Étape 2)
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for document in documents:
        chunks.extend(text_splitter.split_text(document))
    return chunks

# Store Embeddings (Étape 3)
def store_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Utilisation de Hugging Face
    embeddings = model.encode(chunks, convert_to_tensor=True)

    # Déplacer les embeddings sur le CPU avant de les convertir en numpy
    embeddings_np = embeddings.cpu().numpy()

    faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Index pour la recherche
    faiss_index.add(embeddings_np)  # Ajout des embeddings dans l'index

    return faiss_index, embeddings_np

# Retrieve Relevant Documents (Étape 4)
def retrieve_relevant_documents(query, faiss_index, chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Générer l'embedding de la requête
    query_embedding = model.encode([query])

    # Recherche dans FAISS
    results = faiss_index.search(query_embedding, k=5)  # k = 5, par exemple, récupérer les 5 meilleurs résultats
    relevant_chunks = [chunks[i] for i in results[1].flatten()]  # Indices des résultats pertinents

    return relevant_chunks

# Generate Response (Étape 5)
def generate_response(relevant_chunks):
    # Utilisation de Hugging Face pour la génération de texte
    generator = pipeline("text-generation", model="gpt2")  # Vous pouvez choisir un autre modèle si nécessaire
    context = " ".join(relevant_chunks)
    prompt = f"Basé sur les informations suivantes, réponds à la question : {context}"
    response = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
    
    return response

# Exemple d'utilisation
pdf_path = "chatbot-RH/data/FAQ_RH.pdf"
documents = load_documents(pdf_path)
query = "Quel est le processus pour un entretien de recrutement ?"
# Create FAISS index and embeddings
chunks = split_documents(documents)
faiss_index, embeddings_np = store_embeddings(chunks)

try:
    relevant_chunks = retrieve_relevant_documents(query, faiss_index, chunks)
    response = generate_response(relevant_chunks)
    print(response)
finally:
    # Explicitly clean up FAISS resources
    del faiss_index