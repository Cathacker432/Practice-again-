import os
import chromadb
from chromadb.utils import embedding_functions

DATA_FOLDER = "Data"
CHUNK_SIZE = 600 # max characters per chunk


# going through the files and read the text
documents = {}
for file_name in os.listdir(DATA_FOLDER):
    if file_name.endswith(".txt"):
        with open(os.path.join(DATA_FOLDER, file_name), encoding="utf-8") as f:
            documents[file_name] = f.read()
        print("Read", file_name)


# Step 2: split each file into chunks
# I split on empty lines (paragraphs) and join paragraphs until the chunk is big enough
def make_chunks(text):
    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) > CHUNK_SIZE and current:
            chunks.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# Step 3: embed the chunks and save them in Chroma
client = chromadb.PersistentClient(path="chroma_db") # creates a client that saves data to a disk
collection = client.get_or_create_collection("sops") #find the chunck that is similar to the user question

# embedding model (all-MiniLM-L6-v2) - app.py must use the same one for the question
embedder = embedding_functions.DefaultEmbeddingFunction()

for file_name, text in documents.items():
    chunks = make_chunks(text)
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name} for _ in chunks]
    embeddings = embedder(chunks)  # turn each chunk into a vector
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    print(file_name, "->", len(chunks), "chunks")

print("Total chunks saved:", collection.count())
