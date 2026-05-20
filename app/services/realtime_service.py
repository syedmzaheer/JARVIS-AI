"""
REALTIME GROQ SERVICE MODULE
===========================
Extends GroqService to add Tavily web search before calling the LLM. Used by 
ChatService for POST /chat/realtime. Same session and history as general chat; 
the only difference is we run a Tavily search for the user's question and add 
the results to the system message, then call Groq.

ROUND-ROBIN API KEYS:
  - Shares the same round-robin counter as GroqService (class-level _shared_key_index) 
  - This means /chat and /chat/realtime requests use the same rotation sequence 
  - Example: If /chat uses key 1, the next /chat/realtime request will use key 2 
  - All API key usage is logged with masked keys for security and debugging

FLOW:

  1. search_tavily (question): call Tavily API, format results as text (or on failure). 
  2. get_response(question, chat_history): add search results to system message, 
    then same as parent: retrieve context from vector store, build prompt, call Groq.

 IF TAVILY_API_KEY is not set, tavily_client is None and search_tavily returns ""; 
 the user still gets an answer from Groq with no search results.
"""

from typing import List, Optional
from tavily import TavilyClient
import logging
import os

from app.services.groq_service import GroqService, escape_curly_braces
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import JARVIS_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AISystemMessage


logger = logging.getLogger("J.A.R.V.I.S.")


# ===========================================================================================
# REALTIME GROQ SERVICE CLASS  (extends GroqService)
# ===========================================================================================

class RealtimeGroqService(GroqService):
    """
    """