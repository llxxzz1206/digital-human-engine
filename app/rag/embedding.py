from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding 服务抽象层，初始实现使用 OpenAI 兼容 API"""

    def __init__(self) -> None:
        self._provider = settings.milvus.embedding_provider
        self._model = settings.milvus.embedding_model
        self._api_key = settings.llm.api_key
        self._api_base = settings.llm.api_base

    async def embed(self, text: str) -> list[float] | None:
        """将文本向量化

        Args:
            text: 输入文本

        Returns:
            向量列表，失败时返回 None
        """
        if self._provider == "openai_compatible" and self._api_key:
            return await self._openai_embed(text)

        # 无 API Key：返回 None 表示失败（不再用随机向量污染向量库）
        logger.error("Embedding 未配置 API Key，无法向量化")
        return None

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """批量向量化，失败项为 None"""
        return [await self.embed(text) for text in texts]

    async def _openai_embed(self, text: str) -> list[float] | None:
        """使用 OpenAI 兼容 API 进行向量化"""
        base_url = (self._api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
        except Exception as e:
            logger.error("Embedding API 调用失败: %s", e)
            return None


embedding_service = EmbeddingService()
