"""
J.A.R.V.I.S MAIN API
====================

This module defines the FastAPI application and all HTTP endpoints. It is 
designed for single-user use: one person runs one server (e.g. python run.py) 
and uses it as their personal J.A.R.V.I.S backend. Many people can each run 
their own copy of this code on their own machine.

ENDPOINTS: 
  GET                     - Returns API name and list of endpoints.
  GET /health             - Returns status of all services (for monitoring).
  POST /chat              - General chat: pure LLM, no web search. Uses learning data 
                            and past chats via vector-store retrieval only.
  POST /chat/realtime     - Realtime chat: runs a Tavily web search first, then 
                            sends results + context to Grog. Same session as /chat.
  GET /chat/history/{id}  - Returns all messages for a session (general + realtime).

SESSION:
  Both /chat and /chat/realtime use the same session_id. If you omit session_id, 
  the server generates a UUID and returns it; send it back on the next request 
  to continue the conversation. Sessions are saved to disk and survive restarts.

STARTUP:
  On startup, the lifespan function builds the vector store from learning_data/*.txt
  and chats_data/*.json, then creates Groq, Realtime, and Chat services. On shutdown,
  it saves all in-memory sessions to disk.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.models import ChatRequest, ChatResponse

# User-friendly message when Groq rate limit (daily token quota) is exceeded.
RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant."
    "Your credits will reset in a few hours, or you can upgrade your plan for more."
    "Please try again later."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception is a Groq rate limit (429 / tokens per day)"""
    msg = str(exec).lower()
    return "429" in str(exc) or "rate limit" in msg or "token per day" in msg


from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService
from config import VECTOR_STORE_DIR
from langchain_community.vectorstores import FAISS


# ==========================================================================================
# LOGGING
# ==========================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("J.A.R.V.I.S")


# ==========================================================================================
# GLOBAL SERVICE REFERENCES
# ==========================================================================================
# Set during startup (lifespan) and used by all route handlers.
# Stored as globals so async endpoints can access the same service instances. 

vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatService = None

def print_title():
    title = r"""
                     ██╗        █████╗        ██████╗       ██╗   ██╗      ██╗       ███████╗   
                     ██║       ██╔══██╗       ██╔══██╗      ██║   ██║      ██║       ██╔════╝   
                     ██║       ███████║       ██████╔╝      ██║   ██║      ██║       ███████╗   
                ██   ██║       ██╔══██║       ██╔══██╗      ╚██╗ ██╔╝      ██║       ╚════██║   
                ╚█████╔╝  ██╗  ██║  ██║  ██╗  ██║  ██║  ██╗  ╚████╔╝  ██╗  ██║  ██╗  ███████║ ██╗
                 ╚════╝   ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝   ╚═══╝   ╚═╝  ╚═╝  ╚═╝  ╚══════╝ ╚═╝
                                                                     
    """
    print(title)


# ======================================================================================================
# LIFESPAN (STARTUP / SHUTDOWN)
# ======================================================================================================



@asynccontextmanager
async def lifespan (app: FastAPI):
    """
    Application lifespan manager handles startup and shutdown.

    This function manages the application's lifecycle:
    -  STARTUP: Initializes all services in the correct order
      1. VectorStoreService: Creates FAISS index from learning data and chat history 
      2. GroqService: Sets up general chat AI service
      3. RealtimeGroqService: Sets up realtime chat with Tavily search
      4. ChatService: Manages chat sessions and conversations
    - RUNTIME: Application runs normally
    - SHUTDOWN: Saves all active chat sessions to disk

    The services are initialized in this specific order because:
    - VectorStoreService must be created first (used by GroqService)
    - GroqService must be created before RealtimeGroqService (it inherits from it)
    - ChatService needs both GroqService and RealtimeGroqService
    All services are stored as global variables so they can be accessed by API endpoints.
    """

    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()
    logger.info("=" * 60)
    logger.info("J.A.R.V.I.S. - Starting UP...")
    logger.info("=" * 60)


  
    try:
      # Initialize vector store service
      logger.info("Initializing vector store service...")
      vector_store_service = VectorStoreService()
      vector_store_service.create_vector_store()
      logger.info("Vector store initialized successfully")

      # Initialize Groq service (general chat)
      logger.info("Initializing Groq service (general queries)...") 
      groq_service = GroqService(vector_store_service) 
      logger.info("Groq service initialized successfully")

      # Initialize Realtime Groq service (with Tavily search)
      logger.info("Initializing Realtime Groq service (with Tavily search)...") 
      realtime_service = RealtimeGroqService (vector_store_service) 
      logger.info("Realtime Groq service initialized successfully")

      # Initialize chat service
      logger.info("Initializing chat service...")
      chat_service = ChatService (groq_service, realtime_service) 
      logger.info("Chat service initialized successfully")

    
      # Startup complete 
      logger.info("=" * 60) 
      logger.info("Service Status:")
      logger.info("  - Vector Store: Ready")
      logger.info("  - Groq AI (General): Ready")
      logger.info("  - Groq AI (Realtime): Ready")
      logger.info("  - Chat Service: Ready")
      logger.info("=" * 60)
      logger.info("J.A.R.V.I.S is online and ready!") 
      logger.info("API: http://localhost:8000") 
      logger.info("Docs: http://localhost:8000/docs") 
      logger.info("=" * 60)

      yield

      # Shutdown: Save active sessions
      logger.info("\nShutting down J.A.R.V.I.S...")
      if chat_service:
          for session_id in list (chat_service.sessions.keys()):
              chat_service.save_chat_session(session_id)
      logger.info("All sessions saved. Goodbye!")

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise
    

    