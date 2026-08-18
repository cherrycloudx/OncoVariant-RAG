from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

# 1. Point to your existing vector database
db_path = "chroma_db"
embeddings = FakeEmbeddings(size=768)

# 2. Load the database
db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# 3. Ask a test question related to your papers
query = "What are the main mutations and treatment resistances discussed in the papers?"
print(f"\nSearching papers for: '{query}'...\n")

# 4. Search the database for the most relevant chunks
results = db.similarity_search(query, k=3)

# 5. Print out what it found
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(doc.page_content[:400] + "...\n")  # Prints the first 400 characters of each snippet
