import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
CHROMA_PATH = os.path.join("data", "processed", "chroma_db")

def format_docs(docs):
    """Helper function to combine our retrieved chunks into a single readable string."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_pipeline():
    """Builds the modern LCEL RAG pipeline."""
    print("1. Connecting to Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Create the retriever (fetches top 3 chunks)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("2. Booting up Llama 3.2 via Ollama...")
    llm = ChatOllama(model="llama3.2")

    print("3. Assembling the LCEL Chain...")
    
    # The Prompt Template
    template = """You are an expert AI assistant. Use the following pieces of retrieved context to answer the user's question. If the answer is not in the context, clearly state that you do not know. Do not hallucinate or make up information outside the context.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # ---------------------------------------------------------
    # THE LCEL PIPELINE (This is the modern way to write RAG)
    # ---------------------------------------------------------
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Test the pipeline
if __name__ == "__main__":
    # Initialize the entire system
    rag_system = setup_rag_pipeline()
    
    print("\n✅ System Ready. Type 'exit' to quit.\n")
    
    # Create a continuous chat loop in the terminal
    while True:
        user_query = input("\nAsk a question about your PDFs: ")
        
        if user_query.lower() == 'exit':
            break
            
        if not user_query.strip():
            continue
            
        print("\nThinking...\n")
        
        # Invoke the chain with the user's string
        response = rag_system.invoke(user_query)
        
        print("--- ANSWER ---")
        print(response)
        print("-" * 50)