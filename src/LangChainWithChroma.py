import os
from typing import Dict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Load Environment ---
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found.")

# --- Initialize Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.2,
    google_api_key=API_KEY,
)

# --- Connect to ChromaDB ---
CHROMA_DB_DIR = "src/chroma_db"

retriever = None
if os.path.exists(CHROMA_DB_DIR):
    print("✅ Found ChromaDB vector store. Using RAG mode...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
    )
    # Increase k to get more context documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print("⚠️ No ChromaDB found. Falling back to LLM-only mode.")

# --- Memory Store Setup ---
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]

# --- RAG Context Retrieval Function ---
def format_docs(docs):
    """Format retrieved documents into a context string."""
    if not docs:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        # Include metadata if available (like page number)
        metadata = doc.metadata
        page_info = f"[Page {metadata.get('page', 'unknown')}]" if metadata else ""
        context_parts.append(f"Document {i} {page_info}:\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)

def get_context_for_question(inputs: dict) -> str:
    """Retrieve context from RAG if available."""
    if retriever:
        question = inputs.get("question", "")
        docs = retriever.invoke(question)
        return format_docs(docs)
    return ""

# --- Enhanced Prompt with RAG Context ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to specific goverment documents.

IMPORTANT INSTRUCTIONS:
- When context is provided below, use it to answer the question accurately and completely.
- If the context contains the answer, provide it directly without mentioning that you found it in the context.
- Do NOT say "the content is not included" if information is present in the context.
- If you genuinely cannot find the answer in the provided context, then you may say so.


CONTEXT FROM DOCUMENTS:
{context}

Answer the user's question based on the context above."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])

# --- Build the Chain with RAG Integration ---
base_chain = (
    RunnablePassthrough.assign(context=get_context_for_question)
    | prompt 
    | llm 
    | StrOutputParser()
)

# --- Global Chain with History ---
chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    history_messages_key="chat_history",
    input_messages_key="question",
)

__all__ = ["chain_with_history", "retriever"]