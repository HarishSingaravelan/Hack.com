from fastapi import FastAPI
from pydantic import BaseModel
# IMPORTANT: This import connects the FastAPI app to the core
# LangChain logic, including the LLM and the memory store.
from src.LangChainLogic import chain_with_history 

# --- FastAPI Application Setup ---

app = FastAPI(title="LangChain Gemini Conversational API")

# Define the data structure for the incoming request (required by the endpoint)
class QuestionRequest(BaseModel):
    """Input model requiring a session ID for history tracking and the user's question."""
    session_id: str 
    question: str

# Define the data structure for the outgoing response
class AnswerResponse(BaseModel):
    """Output model containing the AI's answer."""
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question_endpoint(request: QuestionRequest):
    """
    Handles user chat requests. It retrieves or creates a session history 
    based on the provided session_id and invokes the LangChain with context.
    """
    
    # Invoke the imported chain_with_history (from llm_service.py).
    # 1. The main input is the user's question dictionary: {"question": request.question}
    # 2. The session context is passed in the 'config' dictionary under "configurable".
    response_text = await chain_with_history.ainvoke(
        {"question": request.question},
        config={"configurable": {"session_id": request.session_id}},
    )
    
    return AnswerResponse(answer=response_text)
