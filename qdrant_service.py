import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantService:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = AsyncQdrantClient(host=host, port=port)
        self.embedder = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.collection_name = "news_embeddings"
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def ensure_collection(self):
        collections = await self.client.get_collections()
        exists = any(
            c.name == self.collection_name for c in collections.collections
        )
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[INFO] Коллекция '{self.collection_name}' создана в Qdrant")

    async def _embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: list(self.embedder.embed(text))[0].tolist(),
        )

    async def add_article(
        self, article_id: int, title: str, content: str, source: str
    ):
        try:
            text = f"{title} {content}"
            vector = await self._embed(text)
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=article_id,
                        vector=vector,
                        payload={
                            "title": title,
                            "content": content,
                            "source": source,
                        },
                    )
                ],
            )
        except Exception as e:
            print(f"[WARN] Qdrant add_article error: {e}")

    async def search(
        self, query: str, limit: int = 5
    ) -> list[dict]:
        try:
            vector = await self._embed(query)
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            return [r.payload for r in response.points]
        except Exception as e:
            print(f"[WARN] Qdrant search error: {e}")
            return []
