import os
from typing import Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory

# --- Environment Setup (API Key Fix) ---

# Load environment variables. This must be done where the LLM is initialized.
load_dotenv()

# Retrieve the API key robustly
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=API_KEY 
)

# --- Session History Mock Store and Factory ---

# In-memory dictionary to store chat histories, mapping session_id to memory object.
# NOTE: For production, replace this in-memory store with Redis or a database.
SESSION_STORE: Dict[str, ConversationSummaryBufferMemory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a ConversationSummaryBufferMemory instance for a given session ID.
    This function is required by RunnableWithMessageHistory.
    """
    if session_id not in SESSION_STORE:
        print(f"Creating new session history for ID: {session_id}")
        
        # Initialize the Summary Buffer Memory.
        SESSION_STORE[session_id] = ConversationSummaryBufferMemory(
            llm=llm, # The LLM used to generate summaries
            max_token_limit=1000, # Max token limit before summarization kicks in
            memory_key="chat_history", 
            return_messages=True
        )
    return SESSION_STORE[session_id].chat_memory

# --- LangChain Chain Setup (with History) ---

# 1. Define the prompt template, including the MessagesPlaceholder for history.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. You must remember the user's name and previous questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# 2. Create the base chain (Prompt -> LLM -> Output Parser)
base_chain = prompt | llm | StrOutputParser()

# 3. Wrap the base chain with history management and export it
chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    history_messages_key="chat_history", 
    input_messages_key="question", 
)
