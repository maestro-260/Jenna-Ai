import chromadb
import uuid
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma")
        self.collection = self.client.get_or_create_collection(
            "context",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_entries = 100000
        self.cleanup_threshold = 0.9  # 90% full

    def store(self, text: str, metadata: dict) -> str:
        """Store new memory with timestamp"""
        doc_id = str(uuid.uuid4())
        full_metadata = {
            **metadata,
            "timestamp": datetime.now().isoformat()
        }
        embedding = self.embedding_model.encode([text])[0].tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[full_metadata],
            ids=[doc_id]
        )
        return doc_id

    def retrieve(self, query: str, n=3, filters=None) -> list:
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            where=filters
        )

        if (
            not results or
            not results.get("documents") or
            not results["documents"][0]
        ):
            return []  # Return empty list instead of crashing

        return [{
            'text': doc,
            'meta': meta,
            'score': score
        } for doc, meta, score in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]

    def bulk_store(self, texts: List[str], metadatas: List[dict]):
        """Store multiple entries efficiently"""
        embeddings = self.embedding_model.encode(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=[str(uuid.uuid4()) for _ in texts]
        )

    async def compact_memory(self):
        """Cleanup old entries when threshold is reached."""
        total_entries = len(self.collection.get()["ids"])
        if total_entries > self.max_entries * self.cleanup_threshold:
            self.logger.info("Running memory compaction...")
            entries = self.collection.get(include=["metadatas", "ids"])
            sorted_entries = sorted(
                zip(entries["metadatas"], entries["ids"]),
                key=lambda x: x[0].get("timestamp", "1970-01-01"),
                reverse=True
            )
            to_keep = sorted_entries[:self.max_entries]
            to_delete = sorted_entries[self.max_entries:]
            if to_delete:
                self.collection.delete([x[1] for x in to_delete])
                self.logger.info(f"Removed {len(to_delete)} old entries")