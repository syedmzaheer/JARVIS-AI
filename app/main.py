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
async def lifespan(app: FastAPI):
    