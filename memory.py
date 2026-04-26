"""
Long-term memory for the agents, backed by ChromaDB.

Each learning agent (bull, bear, trader, invest_judge, risk_manager) gets its
own private ChromaDB collection so retrieved lessons stay relevant to that
agent's role.
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    """Vector-store backed memory of (situation, recommendation) pairs."""

    def __init__(self, name: str, config: dict):
        self.embedding_model = "text-embedding-3-small"
        self.client = OpenAI(base_url=config["backend_url"])

        # allow_reset=True is convenient for notebooks / tests
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.get_or_create_collection(
            name=name
        )

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------
    def get_embedding(self, text: str):
        response = self.client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_situations(self, situations_and_advice):
        """situations_and_advice: list of (situation_text, recommendation_text)."""
        if not situations_and_advice:
            return

        offset = self.situation_collection.count()
        ids = [str(offset + i) for i, _ in enumerate(situations_and_advice)]
        situations = [s for s, _ in situations_and_advice]
        recommendations = [r for _, r in situations_and_advice]
        embeddings = [self.get_embedding(s) for s in situations]

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in recommendations],
            embeddings=embeddings,
            ids=ids,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_memories(self, current_situation: str, n_matches: int = 1):
        """Retrieve the n most-similar past situations' recommendations."""
        if self.situation_collection.count() == 0:
            return []

        query_embedding = self.get_embedding(current_situation)
        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_matches, self.situation_collection.count()),
            include=["metadatas"],
        )
        return [
            {"recommendation": meta["recommendation"]}
            for meta in results["metadatas"][0]
        ]


def build_memories(config: dict):
    """Build one private memory store per learning agent."""
    return {
        "bull":          FinancialSituationMemory("bull_memory", config),
        "bear":          FinancialSituationMemory("bear_memory", config),
        "trader":        FinancialSituationMemory("trader_memory", config),
        "invest_judge":  FinancialSituationMemory("invest_judge_memory", config),
        "risk_manager":  FinancialSituationMemory("risk_manager_memory", config),
    }
