import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify the key is loaded (optional safety check)
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found! Please check your .env file.")

from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 2. LOAD DATABASE & RETRIEVER
db = Chroma(persist_directory="chroma_db", embedding_function=FakeEmbeddings(size=768))
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. INITIALIZE LLM
llm = ChatGroq(model_name="openai/gpt-oss-20b")

# 4. CREATE PROMPT TEMPLATE
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. BUILD MODERN RAG CHAIN
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# 6. INTERACTIVE CHAT LOOP
print("\n--- OncoVariantRAG Chatbot Initialized! ---")
print("Type your questions below. Type 'exit' or 'quit' to stop.\n")

while True:
    question = input("Ask a question: ")
    
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
        
    if not question.strip():
        continue

    print("\nSearching papers and generating answer...\n")
    response = chain.invoke(question)
    
    print("Bot Answer:")
    print(response)
    print("-" * 50 + "\n")

