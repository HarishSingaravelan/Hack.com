import os
from typing import Dict
from dotenv import load_dotenv

# LangChain core imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# RAG Components
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


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

# --- Optional: Connect to ChromaDB ---
CHROMA_DB_DIR = "chroma_db"

rag_chain = None
if os.path.exists(CHROMA_DB_DIR):
    print("✅ Found ChromaDB vector store. Using RAG mode...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Simple retrieval + LLM chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
else:
    print("⚠️ No ChromaDB found. Falling back to LLM-only mode.")


# --- Memory Store Setup (FIXED) ---
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


# --- Prompt Definition ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use context if available, otherwise rely on your own knowledge."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])

# --- Build the Chain ---
if rag_chain:
    # Wrap retrieval into the chain logic
    def get_rag_context(question: str) -> str:
        retrieved = rag_chain({"query": question})
        context_docs = retrieved.get("source_documents", [])
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        return f"Context:\n{context_text}\n\nQuestion: {question}"
else:
    def get_rag_context(question: str) -> str:
        return question

# Base LLM pipeline
base_chain = prompt | llm | StrOutputParser()

# --- Global Chain with History ---
chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    history_messages_key="chat_history",
    input_messages_key="question",
)

# --- Helper function used internally or for testing ---
async def run_with_context(session_id: str, question: str) -> str:
    question_with_context = get_rag_context(question)
    response = await chain_with_history.ainvoke(
        {"question": question_with_context},
        config={"configurable": {"session_id": session_id}},
    )
    return response


__all__ = ["chain_with_history", "run_with_context"]