"""
CHAT SERVICE MODULE
===================

This service owns all chat session and conversation logic. It is used by the
/chat and /chat/realtime endpoints. Designed for single-user use: one server
has one ChatService and one in-memory session store; the user can have many 
sessions (each identified by session_id).

RESPONSIBILITIES:
  - get_or_create_session(session_id): Return existing session or create new one.
    If the user sends a session_idmthat was used before (e.g.- before a restart),
    we try to load it from disk so the conversation continues.
  - add_messages / get_chat_history: Keep messages in memory per session.
  - format_history_for_llm: Trun the message list into (user,assistant) pairs
    and trim to MAX_CHAT_HISTORY_TURNS so we dont overflow the prompt.
  - process_message / process_relatime_message: Add user message, call Groq (or
    RealtimeGroq), add assistant reply, return reply.
  - save_chat_session: Write sessions to database/chat_data/*.json so it persists
    and can be loaded on next startup (and used by the vsctor store for retrival)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from config import CHAT_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService


logger = logging.getLogger("J.A.R.V.I.S.")


# ==========================================================================================================================
# CHAT SERVICE CLASS
# ==========================================================================================================================

class ChatService:
    """
    Manages chat sessions; in-memory message lists, load/save to disk, and
    calling Groq (or Realtime) to get replies. All state for active sessions
    is in self.sessions; saving to disk is done after each messages so
    conversations survive restarts.
    """
    
    def __init__(self, groq_service: GroqService, realtime_service: RealtimeGroqService = None):
        """Store references to the Groq and Realtime services; keep sessions in memory"""
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        #Map: session_id -> list of ChatMessage (user and assitant messages in order)
        self.sessions: Dict[str, List[ChatMessage]] = {}

# ==========================================================================================================================
# SESSION LOAD / VALIDATE / GET-OR-CREATE
# ==========================================================================================================================

    def load_session_from_disk(self, session_id: str) -> bool:
        """
        Load a session from database/chats-data/ if a file for this session_id existis.
        
        File name is chat_{safe_session_id}.json where safe_session_id  has dashes/spaces removed.
        On success we put the messages in self.sessions[session_id] so later requests use them.
        Returns True if loaded, False if file missing or unreadable.
        """
        # Sanitize ID for use in filename (no dashes or spaces)
        safe_session_id = session_id.replace("-", "").replace(" ", "")
        filename = f"chat_{safe_session_id}.json "
        filepath = CHAT_DATA_DIR / filename

        if not filepath.exists():
            return False
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
            # Convert stored dicts back to ChatMessage objects
            messages = [
                ChatMessage(role=msg.get("role"), content=msg.get("content")) 
                for msg in chat_dict["messages"]
            ]
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning("Error loading session %s from disk: %s", session_id, e)
            return False
        
    def validate_session_id(self, session_id: str) -> bool:
        """
        Return True is session_id is safe to use (non-empty, no path traversal, length <= 255).
        Used to reject maliciout or invalid IDs before we use them in file paths.
        """
        if not session_id or not session_id.strip():
            return False
        # Block path traversal and path separators.
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 225:
            return False
        return True
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Return"""