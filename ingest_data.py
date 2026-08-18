import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

# 1. Point to your data folder
DATA_PATH = "data/"
db_path = "chroma_db"

# Initialize a simple embedding model (We use a fake one for now, we'll upgrade later)
embeddings = FakeEmbeddings(size=768)

# 2. Loop through all PDFs in the folder
all_docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        print(f"Loading {file}...")
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        all_docs.extend(loader.load())

# 3. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(all_docs)

# 4. Save to ChromaDB
print(f"Saving {len(chunks)} chunks to vector database...")
db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)

print("Success! Your research papers are now indexed and searchable.")
