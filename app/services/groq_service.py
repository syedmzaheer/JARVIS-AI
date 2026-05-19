"""
GROQ SERVICE MODULE
===================

This module handles general chat: no web search, only the Groq LLM plus context 
from the vector store (learning data past chats). Used by ChatService 
for POST /chat.

MULTIPLE API KEYS (round-robin and fallback):
  - You can set multiple Groq API keys in env: GROQ API KEY, GROQ API_KEY_2,
  GROQ API KEY 3,... (no limit).
  - Each request uses one key in rotation: 1st request -> 1st key, 2nd request ->
    2nd key, 3rd request -> 3rd key, then back to 1st key, and so on. Every key
    is used one-by-one so usage is spread across keys.
  - The round-robin counter is shared across all instances (GroqService and 
    RealtimeGroqService), so both /chat and/chat/realtime endpoints use the 
    same rotation sequence.
  - If the chosen key fails (rate limit 429 or any error), we try the next key,
    then the next, until one succeeds or all have been tried.
  - All API key usage is logged with masked keys (first 8 and last 4 chars visible)
    for security and debugging purposes.

FLOW:
  1. get response(question, chat history) is called.
  2. We ask the vector store for the top-k chunks most similar to the question (retrieval).
  3. Me build a system message: JARVIS_SYSTEM PROMPT current time retrieved context.
  4. We send to Groq using the next key in rotation (or fallback to next key on failure).
  5. He return the assistant's reply.

Context is only what we retrieve (no full dump of learning data), token usage stays bounded.
"""

from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

from config import GROQ_API_KEYS, GROQ_MODEL, JARVIS_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S.")


# ==========================================================================================================================
# HELPER: ESCAPE CURLY BRACES FOR LANGCHAIN
# ==========================================================================================================================
# LangChain prompt templates use {variable_name}. If learning data or chat
# 
