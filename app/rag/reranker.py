from __future__ import annotations

import asyncio
import logging

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker 重排器，基于阿里 DashScope qwen3-rerank API

    对 RAG 检索结果进行精排，返回按相关性排序的结果。
    使用双阈值 + 来源感知决定路由：
      - source_type="faq" + rerank_score >= threshold_a → faq_direct（零 LLM 调用）
      - source_type="knowledge" + 任意分数 → rag_chat 或 chat（知识库永远不走 direct）
      - rerank_score < threshold_b → chat（普通对话）
    """

    def __init__(self) -> None:
        self._model = settings.rag.rerank_model
        self._threshold_a = settings.rag.rerank_threshold_a
        self._threshold_b = settings.rag.rerank_threshold_b
        self._api_key = settings.llm.api_key
        # DashScope 原生 rerank API 端点
        self._rerank_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """对检索结果重排

        Args:
            query: 用户查询
            documents: RAG 检索结果列表，每个包含 "text", "score" 等
            top_k: 返回前 K 个结果

        Returns:
            重排后的结果列表，按 rerank_score 降序，每个结果新增 "rerank_score" 字段
        """
        if not documents:
            return []

        if not self._api_key:
            logger.warning("Reranker: 无 API Key，跳过重排，使用原始分数")
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0.0)
            return documents[:top_k]

        try:
            results = await self._api_rerank(query, documents, top_k)
            return results
        except Exception as e:
            logger.error("Reranker API 调用失败: %s，回退到原始分数", e)
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0.0)
            return documents[:top_k]

    async def _api_rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]:
        """调用 DashScope rerank API（原生格式）"""
        texts = [doc.get("text", "") for doc in documents]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self._rerank_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": {
                        "query": query,
                        "documents": texts,
                    },
                    "parameters": {
                        "top_n": min(top_k, len(documents)),
                        "return_documents": False,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        # DashScope 原生格式：output.results 列表，包含 index 和 relevance_score
        results_data = data.get("output", {}).get("results", [])
        for item in results_data:
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            documents[idx]["rerank_score"] = float(score)

        # 未被 API 返回的文档，分数设为 0
        returned_indices = {item.get("index", -1) for item in results_data}
        for i, doc in enumerate(documents):
            if i not in returned_indices:
                doc["rerank_score"] = 0.0

        # 按 rerank_score 降序排序
        documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return documents[:top_k]

    def route(self, max_rerank_score: float, source_type: str = "knowledge") -> str:
        """根据重排最高分和结果来源决定路由

        Args:
            max_rerank_score: Reranker 最高分
            source_type: "faq" | "knowledge"

        Returns:
            "faq_direct" — FAQ 高相关，零 LLM 调用
            "rag_chat" — 知识库中/高相关，LLM 生成自然回答
            "chat" — 低相关，普通对话
        """
        # 只有 FAQ 来源且高分才能走 direct
        if source_type == "faq" and max_rerank_score >= self._threshold_a:
            return "faq_direct"

        # 知识库永远不走 direct，中/高分走 rag_chat
        if max_rerank_score >= self._threshold_b:
            return "rag_chat"
        else:
            return "chat"

    @property
    def threshold_a(self) -> float:
        return self._threshold_a

    @property
    def threshold_b(self) -> float:
        return self._threshold_b


reranker = Reranker()
