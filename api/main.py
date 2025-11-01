from fastapi import FastAPI
from pydantic import BaseModel
from src.LangChainWithChroma import chain_with_history, retriever

# --- FastAPI Application Setup ---

app = FastAPI(title="LangChain Gemini Conversational API")

class QuestionRequest(BaseModel):
    """Input model requiring a session ID for history tracking and the user's question."""
    session_id: str 
    question: str

class AnswerResponse(BaseModel):
    """Output model containing the AI's answer."""
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question_endpoint(request: QuestionRequest):
    """
    Handles user chat requests with automatic RAG context integration.
    The chain_with_history now automatically retrieves relevant context
    from ChromaDB and includes it in the LLM prompt.
    """
    response_text = await chain_with_history.ainvoke(
        {"question": request.question},
        config={"configurable": {"session_id": request.session_id}},
    )
    
    return AnswerResponse(answer=response_text)

@app.get("/test-rag")
async def test_rag():
    """Debug endpoint to verify RAG is working"""
    if retriever is None:
        return {
            "status": "❌ RAG not enabled",
            "reason": "ChromaDB not found at src/chroma_db"
        }
    
    test_query = "change of address"
    docs = retriever.invoke(test_query)
    
    return {
        "status": "✅ RAG enabled and working!",
        "chroma_dir": "src/chroma_db",
        "test_query": test_query,
        "num_docs_retrieved": len(docs),
        "sample_docs": [
            {
                "page": doc.metadata.get("page", "unknown"),
                "preview": doc.page_content[:200] + "..."
            }
            for doc in docs[:3]
        ] if docs else []
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "rag_enabled": retriever is not None
    }