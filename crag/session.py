"""
Session Management for CRAG Chatbot.

Provides conversation session handling with automatic storage backend detection:
- In-memory storage for local development (no dependencies)
- Redis storage for production (when REDIS_URL is set)

Usage:
    from crag import SessionManager
    
    # Auto-detects storage (Redis if REDIS_URL set, else in-memory)
    manager = SessionManager()
    
    # Get or create a session
    session = manager.get_or_create(session_id)
    
    # Use session history with CRAG
    result = run_crag(question, retriever, history=session.get_history())
    
    # Update session after response
    session.add_turn(question, result["generation"])
    manager.save_session(session)
"""

import os
import uuid
import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============== Configuration ==============

DEFAULT_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
DEFAULT_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))


# ============== Session Data Class ==============

@dataclass
class Session:
    """
    A conversation session holding history and metadata.
    
    Attributes:
        id: Unique session identifier (UUID)
        history: List of conversation messages [{"role": "user"|"assistant", "content": "..."}]
        created_at: When the session was created
        last_activity: When the session was last used
        metadata: Optional user info, preferences, etc.
        max_turns: Maximum conversation turns to keep (older messages are trimmed)
    """
    id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_turns: int = DEFAULT_MAX_TURNS
    
    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Add a conversation turn (user message + assistant response).
        
        Automatically trims history if it exceeds max_turns.
        Updates last_activity timestamp.
        
        Args:
            user_message: The user's question
            assistant_message: The assistant's response
        """
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        self.last_activity = datetime.now()
        
        # Trim history if exceeds max_turns (each turn = 2 messages)
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history in the format expected by CRAG.
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        return self.history.copy()
    
    def is_expired(self, timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> bool:
        """
        Check if the session has expired due to inactivity.
        
        Args:
            timeout_minutes: Minutes of inactivity before expiry
            
        Returns:
            True if session has expired, False otherwise
        """
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def turn_count(self) -> int:
        """Get the number of conversation turns (pairs of user/assistant messages)."""
        return len(self.history) // 2
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.history = []
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize session to dictionary (for storage).
        
        Returns:
            Dictionary representation of the session
        """
        return {
            "id": self.id,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "max_turns": self.max_turns,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """
        Deserialize session from dictionary.
        
        Args:
            data: Dictionary representation of a session
            
        Returns:
            Session instance
        """
        return cls(
            id=data["id"],
            history=data.get("history", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            metadata=data.get("metadata", {}),
            max_turns=data.get("max_turns", DEFAULT_MAX_TURNS),
        )
    
    def __repr__(self) -> str:
        return f"Session(id={self.id[:8]}..., turns={self.turn_count()}, last_activity={self.last_activity.strftime('%H:%M:%S')})"


# ============== Session Store Interface ==============

class SessionStore(ABC):
    """
    Abstract base class for session storage backends.
    
    Implement this interface to add new storage backends (e.g., PostgreSQL, DynamoDB).
    """
    
    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session if found and not expired, None otherwise
        """
        pass
    
    @abstractmethod
    def save(self, session: Session) -> None:
        """
        Save or update a session.
        
        Args:
            session: The session to save
        """
        pass
    
    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def cleanup_expired(self, timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> int:
        """
        Remove all expired sessions.
        
        Args:
            timeout_minutes: Minutes of inactivity before expiry
            
        Returns:
            Number of sessions removed
        """
        pass


# ============== In-Memory Storage ==============

class InMemorySessionStore(SessionStore):
    """
    In-memory session storage using a Python dictionary.
    
    Thread-safe implementation suitable for single-server deployments.
    Sessions are lost when the process restarts.
    
    Best for:
    - Local development
    - Single-server deployments
    - Testing
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
    
    def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID (thread-safe)."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def save(self, session: Session) -> None:
        """Save or update a session (thread-safe)."""
        with self._lock:
            self._sessions[session.id] = session
    
    def delete(self, session_id: str) -> bool:
        """Delete a session (thread-safe)."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def cleanup_expired(self, timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> int:
        """Remove all expired sessions (thread-safe)."""
        with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired(timeout_minutes)
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)
    
    def count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def __repr__(self) -> str:
        return f"InMemorySessionStore(sessions={self.count()})"


# ============== Redis Storage ==============

class RedisSessionStore(SessionStore):
    """
    Redis-based session storage for production deployments.
    
    Uses Redis TTL for automatic session expiration.
    Suitable for multi-server deployments with shared state.
    
    Requires:
    - redis-py package
    - REDIS_URL environment variable or explicit redis_url parameter
    
    Best for:
    - Production deployments
    - Multi-server / load-balanced setups
    - Railway, Heroku, or any cloud with Redis
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        key_prefix: str = "crag:session:",
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize Redis session store.
        
        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            key_prefix: Prefix for Redis keys (for namespacing)
            ttl_seconds: Time-to-live for sessions (defaults to SESSION_TIMEOUT_MINUTES * 60)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        if not self.redis_url:
            raise ValueError(
                "Redis URL not provided. Set REDIS_URL environment variable "
                "or pass redis_url parameter."
            )
        
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or (DEFAULT_TIMEOUT_MINUTES * 60)
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            try:
                import redis
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisSessionStore. "
                    "Install it with: pip install redis"
                )
            self._client = redis.from_url(self.redis_url)
        return self._client
    
    def _make_key(self, session_id: str) -> str:
        """Create a Redis key for a session ID."""
        return f"{self.key_prefix}{session_id}"
    
    def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session from Redis."""
        client = self._get_client()
        key = self._make_key(session_id)
        
        data = client.get(key)
        if data is None:
            return None
        
        try:
            session_dict = json.loads(data)
            session = Session.from_dict(session_dict)
            
            # Check if expired (belt-and-suspenders with Redis TTL)
            if session.is_expired(self.ttl_seconds // 60):
                self.delete(session_id)
                return None
            
            return session
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted session data - delete it
            self.delete(session_id)
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to Redis with TTL."""
        client = self._get_client()
        key = self._make_key(session.id)
        
        data = json.dumps(session.to_dict())
        client.setex(key, self.ttl_seconds, data)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session from Redis."""
        client = self._get_client()
        key = self._make_key(session_id)
        return client.delete(key) > 0
    
    def cleanup_expired(self, timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> int:
        """
        Remove expired sessions.
        
        Note: Redis TTL handles expiration automatically, so this is mostly a no-op.
        It's implemented for interface compatibility.
        """
        # Redis TTL handles expiration automatically
        # This method exists for interface compatibility
        return 0
    
    def __repr__(self) -> str:
        return f"RedisSessionStore(url={self.redis_url[:20]}...)"


# ============== Session Manager ==============

class SessionManager:
    """
    High-level session management API.
    
    Automatically selects storage backend:
    - Redis if REDIS_URL environment variable is set
    - In-memory otherwise
    
    Usage:
        manager = SessionManager()
        
        # Create a new session
        session = manager.create_session()
        
        # Get existing or create new
        session = manager.get_or_create(session_id)
        
        # Update after a conversation turn
        session.add_turn("What is GPT-4?", "GPT-4 is...")
        manager.save_session(session)
    """
    
    def __init__(
        self,
        store: Optional[SessionStore] = None,
        timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
        max_turns: int = DEFAULT_MAX_TURNS,
    ):
        """
        Initialize the session manager.
        
        Args:
            store: Explicit storage backend (auto-detects if None)
            timeout_minutes: Session timeout in minutes
            max_turns: Maximum conversation turns to keep per session
        """
        self.timeout_minutes = timeout_minutes
        self.max_turns = max_turns
        
        if store is not None:
            self.store = store
        elif os.getenv("REDIS_URL"):
            self.store = RedisSessionStore()
        else:
            self.store = InMemorySessionStore()
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Create a new session.
        
        Args:
            metadata: Optional metadata to attach to the session
            
        Returns:
            A new Session instance (already saved to store)
        """
        session = Session(
            id=str(uuid.uuid4()),
            history=[],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata=metadata or {},
            max_turns=self.max_turns,
        )
        self.store.save(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get an existing session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session if found and not expired, None otherwise
        """
        session = self.store.get(session_id)
        
        if session is None:
            return None
        
        # Check expiration
        if session.is_expired(self.timeout_minutes):
            self.store.delete(session_id)
            return None
        
        return session
    
    def get_or_create(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID to look up
            metadata: Optional metadata for new sessions
            
        Returns:
            Existing session if found and valid, new session otherwise
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(metadata)
    
    def save_session(self, session: Session) -> None:
        """
        Save a session to the store.
        
        Call this after modifying session state (e.g., after add_turn).
        
        Args:
            session: The session to save
        """
        self.store.save(session)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        return self.store.delete(session_id)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        return self.store.cleanup_expired(self.timeout_minutes)
    
    def get_storage_type(self) -> str:
        """Get the name of the storage backend being used."""
        return type(self.store).__name__
    
    def __repr__(self) -> str:
        return f"SessionManager(store={self.store}, timeout={self.timeout_minutes}min, max_turns={self.max_turns})"


# ============== Convenience Functions ==============

def create_session_manager(
    redis_url: Optional[str] = None,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> SessionManager:
    """
    Create a SessionManager with explicit configuration.
    
    Args:
        redis_url: Redis URL (uses in-memory if None and REDIS_URL not set)
        timeout_minutes: Session timeout in minutes
        max_turns: Maximum conversation turns to keep
        
    Returns:
        Configured SessionManager instance
    """
    if redis_url:
        store = RedisSessionStore(redis_url=redis_url)
    elif os.getenv("REDIS_URL"):
        store = RedisSessionStore()
    else:
        store = InMemorySessionStore()
    
    return SessionManager(
        store=store,
        timeout_minutes=timeout_minutes,
        max_turns=max_turns,
    )
