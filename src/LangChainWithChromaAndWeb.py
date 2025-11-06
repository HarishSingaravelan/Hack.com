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
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Load Environment ---
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found.")
if not TAVILY_API_KEY:
    print("⚠️ TAVILY_API_KEY not found. Web search will be disabled.")

# --- Initialize Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=API_KEY,
)

# --- Initialize Tavily Search ---
tavily_search = None
if TAVILY_API_KEY:
    try:
        tavily_search = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
        print("✅ Tavily web search enabled.")
    except Exception as e:
        print(f"⚠️ Failed to initialize Tavily: {e}")

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
        return ""
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        page_info = f"[Page {metadata.get('page', 'unknown')}]" if metadata else ""
        context_parts.append(f"Document {i} {page_info}:\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)

def format_web_results(results):
    """Format web search results into a context string."""
    if not results:
        return ""
    
    web_parts = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        content = result.get('content', result.get('snippet', 'No content'))
        url = result.get('url', '')
        web_parts.append(f"Web Result {i}: {title}\nSource: {url}\n{content}")
    
    return "\n\n---\n\n".join(web_parts)

def should_use_web_search(question: str) -> bool:
    """Determine if web search should be used based on question keywords."""
    web_search_keywords = [
        'current', 'latest', 'recent', 'today', 'now', 'news',
        'update', '2024', '2025', 'this year', 'what is happening',
        'weather', 'stock', 'price'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in web_search_keywords)

def get_context_for_question(inputs: dict) -> str:
    """Retrieve context from RAG and optionally web search."""
    question = inputs.get("question", "")
    contexts = []
    
    # 1. Try RAG first (government documents)
    if retriever:
        try:
            docs = retriever.invoke(question)
            rag_context = format_docs(docs)
            if rag_context:
                contexts.append(f"📚 GOVERNMENT DOCUMENTS:\n{rag_context}")
        except Exception as e:
            print(f"RAG retrieval error: {e}")
    
    # 2. Use web search if needed and available
    if tavily_search and should_use_web_search(question):
        try:
            web_results = tavily_search.invoke({"query": question})
            web_context = format_web_results(web_results)
            if web_context:
                contexts.append(f"🌐 WEB SEARCH RESULTS:\n{web_context}")
        except Exception as e:
            print(f"Web search error: {e}")
    
    # 3. Combine all contexts
    if contexts:
        return "\n\n═══════════════════\n\n".join(contexts)
    
    return "No relevant context found."

# --- Enhanced Prompt with RAG and Web Search Context ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to government documents and web search capabilities.

IMPORTANT INSTRUCTIONS:
- When context is provided below, use it to answer the question accurately and completely.
- Government documents are authoritative for official procedures and regulations.
- Web search results provide current, real-time information.
- If the context contains the answer, provide it directly without mentioning that you found it in the context.
- Do NOT say "the content is not included" if information is present in the context.
- If you genuinely cannot find the answer in the provided context, then you may say so.
- Be empathetic towards the user and provide step-by-step instructions instead of dumping information.
- When using web sources, you may cite the source URL if relevant.
- Be empathetic towards the user and try to help them by providing step by step instructions instead of dumping informations.
- If you are given a different language answer in the same language.
- If you cannot get any info on web as well as chromadb you SHOULD  answer using your own knowledge

CONTEXT:
{context}

Answer the user's question based on the context above."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])

# --- Build the Chain with RAG and Web Search Integration ---
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

__all__ = ["chain_with_history", "retriever", "tavily_search"]