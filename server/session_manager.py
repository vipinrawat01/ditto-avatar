###############################################################################
#  Global Session Manager
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar


class MaxSessionError(Exception):
    """Raised when the session count reaches its limit"""
    pass

def _rand_session_id() -> str:
    """Generate a UUID session ID"""
    return str(uuid.uuid4())

class SessionManager:
    """
    Global digital-human session manager.

    Manages the avatar_sessions lifecycle in one place and keeps the service
    available even when detached from WebRTC.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.sessions: Dict[str, BaseAvatar] = {}
            self.build_session_fn = None
            self.max_session = 1   # default, override via set_max_session()
            self.initialized = True

    def set_max_session(self, n: int):
        """Set the maximum number of concurrent sessions"""
        self.max_session = max(1, n)

    def init_builder(self, build_session_fn):
        """Configure the factory function used to build an avatar_session"""
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """Get a live session"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """Check whether a session exists"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None
        
    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        Create a new session in an async environment.
        If sessionid is None, one is generated automatically.
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")
            
        if sessionid is None:
            sessionid = _rand_session_id()
            
        # Check whether the maximum session count has been reached
        active_count = sum(1 for s in self.sessions.values() if s is not None)
        if active_count >= self.max_session:
            raise MaxSessionError(
                f"Maximum session limit reached ({active_count}/{self.max_session})"
            )

        logger.info('Creating sessionid=%s, current session num=%d', sessionid, active_count)
        # Reserve the slot in advance to prevent duplicates
        self.sessions[sessionid] = None

        # Build the session in a thread pool (loading the model is very expensive)
        avatar_session = await asyncio.get_event_loop().run_in_executor(
            None, self.build_session_fn, sessionid, params
        )
        self.sessions[sessionid] = avatar_session
        return sessionid
        
    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """Synchronously add a static or externally managed session (for non-server entry points)"""
        self.sessions[sessionid] = avatar_session
        
    def remove_session(self, sessionid: str):
        """Destroy session resources"""
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            # todo: could also actively call avatar_session to release resources
            self.sessions.pop(sessionid, None)

# Singleton instance
session_manager = SessionManager()
