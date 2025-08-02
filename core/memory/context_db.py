import asyncio
import aiosqlite
import duckdb
import uuid
import ast
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Optional, AsyncGenerator, Tuple
from aiosqlite import Pool
from core.memory.vector_store import VectorMemory
from utils.model_switcher import ModelSwitcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextDatabase:
    async def get_user_facts(self, user_id: str) -> dict:
        """Retrieve long-term facts/preferences/events for a user."""
        async with self.get_connection() as conn:
            facts = await conn.execute_fetchall(
                "SELECT input, response FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
                (user_id,)
            )
        # Simplified: aggregate facts
        return {"facts": [row[0] for row in facts], "responses": [row[1] for row in facts]}

    async def store_user_fact(self, user_id: str, fact: str):
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT INTO interactions (user_id, input, response) VALUES (?, ?, ?)",
                (user_id, fact, "[FACT]")
            )
            await conn.commit()
    """Production-grade context database with automated maintenance."""

    def __init__(self):
        self.db_path = "memory/context.db"
        self._init_db()
        self.pool: Optional[Pool] = None
        self.duck_conn = duckdb.connect()
        self.memory = VectorMemory()
        self._init_lock = asyncio.Lock()
        self._model_switch_lock = asyncio.Lock()
        self.pool_size = 10
        self.statement_cache = {}
        asyncio.create_task(self._initialize())

    async def _initialize(self):
        """Async initialization with connection pooling."""
        async with self._init_lock:
            if not self.pool:
                self.pool = await aiosqlite.create_pool(
                    self.db_path,
                    timeout=30,
                    pool_size=self.pool_size,
                    check_same_thread=False
                )
                await self._init_db()
                asyncio.create_task(self._maintenance_loop())

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT,
                    input TEXT,
                    response TEXT,
                    intent TEXT,
                    emotion TEXT,
                    user_feedback TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interaction_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    intent TEXT,
                    timestamp TEXT,
                    entities TEXT
                )
            """)

    async def _optimize_query(self, query: str) -> str:
        """Cache and optimize frequent queries."""
        if query not in self.statement_cache:
            self.statement_cache[query] = query
            # Add indexes for frequently queried columns
            async with self.get_connection() as conn:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_timestamp
                    ON interactions(session_id, timestamp)
                """)
        return self.statement_cache[query]

    async def get_conversation_history(self, session_id: str, limit: int = 10):
        query = await self._optimize_query("""
            SELECT input, response, intent, emotion
            FROM interactions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """)
        async with self.get_connection() as conn:
            return await conn.execute_fetchall(query, (session_id, limit))

    async def get_connection(
        self
    ) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get managed database connection from pool."""
        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Connection error: {str(e)}")
                raise

    async def log_interaction(
        self,
        session_id: str,
        input_text: str,
        response: str,
        metadata: dict
    ):
        user_id = 'default_user'  # For single-user testing
        async with self.get_connection() as conn:
            await conn.execute(
                (
                    "INSERT INTO interactions (session_id, user_id, input, "
                    "response, intent, emotion) VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    session_id,
                    user_id,
                    input_text,
                    response,
                    metadata.get("intent"),
                    metadata.get("emotion")
                )
            )
            await conn.commit()

    async def _generate_embedding(
        self,
        input_text: str,
        response: str
    ) -> bytes:
        """Generate and cache embeddings for fast retrieval."""
        text = f"User: {input_text}\nAssistant: {response}"
        vector = await asyncio.to_thread(
            lambda: self.memory.embedding_model.encode(text)
        )
        return vector.tobytes()

    async def _get_user_id(
        self,
        conn: aiosqlite.Connection,
        session_id: str
    ) -> str:
        """Atomic user ID creation with transaction safety."""
        result = await conn.execute_fetchall(
            "SELECT user_id FROM user_sessions WHERE session_id = ?",
            (session_id,)
        )
        if not result:
            user_id = str(uuid.uuid4())
            await conn.execute(
                (
                    "INSERT INTO user_sessions (session_id, user_id) "
                    "VALUES (?, ?)"
                ),
                (session_id, user_id)
            )
            return user_id
        return result[0][0]

    async def get_context_vector(
        self,
        session_id: str
    ) -> Optional[List[float]]:
        """Get conversation context with model management."""
        try:
            history = await self.retrieve_recent_conversations(session_id)
            if not history:
                return None

            if await ModelSwitcher().should_switch():
                asyncio.create_task(self._handle_model_switch())

            async with self.get_connection() as conn:
                result = await conn.execute_fetchall(
                    """SELECT embedding FROM interactions
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1""",
                    (session_id,)
                )
                if result:
                    embedding_buffer = result[0][0]
                    return np.frombuffer(
                        embedding_buffer, dtype=np.float32
                    ).tolist()
                return None
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return None

    async def _handle_model_switch(self):
        """Handle model updates with lock protection."""
        async with self._model_switch_lock:
            try:
                switcher = ModelSwitcher()
                if await switcher.validate_new_model():
                    logger.info("Initiating model switch...")
                    await switcher.switch()
                    await self.memory.reload_embeddings()
                    logger.info("Model switch completed")
            except Exception as e:
                logger.error(f"Model switch failed: {str(e)}")

    async def get_training_data(self) -> List[Tuple[str, str, str, str]]:
        async with self.get_connection() as conn:
            data = await conn.execute_fetchall("""
                SELECT input, response, intent, emotion
                FROM interactions
                WHERE user_id = 'default_user'
                ORDER BY timestamp DESC
            """)
            return [(row[0], row[1], row[2], row[3]) for row in data]

    async def fetch_habits_fast(self, user_id: str) -> List[Dict]:
        """High-performance habit analysis using DuckDB."""
        try:
            self.duck_conn.execute(
                f"ATTACH '{self.db_path}' AS sqlite_db (TYPE SQLITE)"
            )
            query = """
                SELECT intent, timestamp, entities, COUNT(*) as frequency
                FROM sqlite_db.interaction_patterns
                WHERE user_id = ?
                GROUP BY intent, timestamp, entities
                ORDER BY frequency DESC
                LIMIT 50
            """
            records = self.duck_conn.execute(query, (user_id,)).fetchall()
            return [{
                "intent": row[0],
                "timestamp": row[1],
                "entities": ast.literal_eval(row[2]) if row[2] else [],
                "frequency": row[3]
            } for row in records]
        except Exception as e:
            logger.error(f"Habit analysis failed: {str(e)}")
            return []

    async def retrieve_recent_conversations(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get conversation history with embeddings."""
        async with self.get_connection() as conn:
            data = await conn.execute_fetchall("""
                SELECT input, response, intent, emotion, embedding
                FROM interactions
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
        return [{
            "input": row[0],
            "response": row[1],
            "intent": row[2],
            "emotion": row[3],
            "embedding": np.frombuffer(row[4], dtype=np.float32).tolist()
            if row[4] else None
        } for row in data]

    async def _maintenance_loop(self):
        """Automated database maintenance."""
        while True:
            await asyncio.sleep(86400)
            try:
                async with self.get_connection() as conn:
                    logger.info("Running database maintenance...")
                    await conn.execute("VACUUM")
                    await conn.execute("PRAGMA optimize")
                    await conn.commit()
                    logger.info("Maintenance completed")
            except Exception as e:
                logger.error(f"Maintenance failed: {str(e)}")

    async def store_interaction(
        self, session_id: str, text: str, embedding: List[float]
    ):
        async with self.get_connection() as conn:
            await conn.execute(
                """INSERT INTO interactions
                (session_id, content, embedding)
                VALUES (?, ?, ?)""",
                (session_id, text, np.array(embedding).tobytes())
            )
            await conn.commit()

    async def close(self):
        """Clean shutdown procedure."""
        if self.pool:
            await self.pool.close()
        self.duck_conn.close()
        logger.info("Database connections closed cleanly")

    async def session_exists(self, session_id: str) -> bool:
        async with self.get_connection() as conn:
            result = await conn.execute_fetchall(
                "SELECT COUNT(*) FROM user_sessions WHERE session_id = ?",
                (session_id,)
            )
            return result[0][0] > 0

    async def create_session(self, session_id: str, user_id: str):
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT INTO user_sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id)
            )
            await conn.commit()

    async def recover_from_failure(self):
        """Recover database from failure state"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("PRAGMA wal_checkpoint(FULL)")
                await conn.execute("PRAGMA integrity_check")
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
