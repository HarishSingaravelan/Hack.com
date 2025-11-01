import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# === Configuration ===
PDF_DIR = "data"            # Folder containing PDFs
CHROMA_DB_DIR = "chroma_db" # Output directory for the vector store
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_chroma_db():
    # Collect all PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF files found in 'data/' directory.")
        return

    print(f"📚 Found {len(pdf_files)} PDFs. Loading and splitting...")

    all_docs = []
    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Split text into manageable chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_docs)

    print(f"✅ Created {len(split_docs)} text chunks.")

    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create and persist Chroma database
    print("💾 Building and saving ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    vectorstore.persist()
    print(f"✅ ChromaDB successfully created at: {CHROMA_DB_DIR}")

if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    build_chroma_db()
