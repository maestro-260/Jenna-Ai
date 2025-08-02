import ollama
import sqlite3
import duckdb
import pandas as pd
import cupy as cp  # GPU acceleration
from cuml.cluster import KMeans  # GPU-based clustering
from datetime import datetime
from typing import List, Dict
from core.memory.vector_store import VectorMemory


class SmartHabitAI:
    def __init__(self, db_path="memory/habits.db"):
        self.db_path = db_path
        self.memory = VectorMemory()
        self.duck_conn = duckdb.connect()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS habits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    habit_name TEXT,
                    frequency INTEGER
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

    def log_interaction(self, user_id: str, intent: str, entities: list):
        """Logs user interactions into the habit database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO interaction_patterns (user_id, intent, timestamp,
                entities) VALUES (?, ?, ?, ?)
                """,
                (user_id, intent, datetime.now().isoformat(), str(entities))
            )

    async def predict_suggestions(self, user_id: str):
        """
        Uses Ollama to predict personalized suggestions from past
        interactions.
        """
        habits = self.memory.retrieve(f"user_habits_{user_id}", n=5)
        prompt = (
            f"The user has these behavior patterns: {habits}. "
            "What would be the best proactive suggestion for them?"
        )
        try:
            response = await ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.get("message", "No suggestions found.")
        except Exception as e:
            print(f"Ollama request failed: {e}")
            return "I couldn't generate a suggestion at this time."

    def analyze_habits_gpu(self, user_id: str):
        """
        Uses GPU acceleration for clustering user habits based on
        active times.
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                (
                    "SELECT timestamp FROM interaction_patterns "
                    "WHERE user_id = ?"
                ),
                conn, params=(user_id,)
            )

        if df.empty:
            return {
                "active_hours": [8, 12, 18]
            }  # Default morning, noon, evening

        hours = cp.array(
            [datetime.fromisoformat(ts).hour for ts in df['timestamp']]
        )

        if len(hours) < 3:
            return {"active_hours": [int(cp.mean(hours))]}
        # Default to average

        kmeans = KMeans(n_clusters=3, random_state=0).fit(hours.reshape(-1, 1))
        clusters = cp.asnumpy(kmeans.cluster_centers_).tolist()

        return {'active_hours': sorted([round(c) for c in clusters])}


class ProactiveSuggestor:
    def __init__(self):
        self.habit_ai = SmartHabitAI()

    async def generate_suggestions(self, user_id: str) -> List[Dict]:
        """Generates proactive suggestions based on AI habit predictions."""
        stats = self.habit_ai.analyze_habits_gpu(user_id)
        suggestion_text = await self.habit_ai.predict_suggestions(user_id)

        return [{
            'message': suggestion_text,
            'type': "smart_prediction",
            'action': "personalized_recommendation",
            'metadata': stats
        }]