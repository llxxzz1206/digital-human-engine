"""reranker.route 双阈值路由单元测试（项4）

锁住路由不变量（对应 docs/dev-improvement-plan.md 项4）：
- 只有 faq 来源 + score >= threshold_a 才能走 faq_direct（零 LLM 调用）
- knowledge 来源永远不走 direct，中/高分走 rag_chat
- score < threshold_b → chat 普通对话
- 无 API Key / API 异常时降级用原始分，不抛异常
"""

import pytest

from app.rag.reranker import Reranker


@pytest.fixture
def rk() -> Reranker:
    return Reranker()


class TestRoute:
    def test_faq_high_score_goes_direct(self, rk: Reranker):
        assert rk.route(0.95, "faq") == "faq_direct"

    def test_faq_boundary_a_goes_direct(self, rk: Reranker):
        # >= threshold_a 含等号：恰好等于 a 也走 direct
        assert rk.route(rk.threshold_a, "faq") == "faq_direct"

    def test_faq_mid_score_goes_rag_chat(self, rk: Reranker):
        mid = (rk.threshold_a + rk.threshold_b) / 2
        assert rk.route(mid, "faq") == "rag_chat"

    def test_faq_low_score_goes_chat(self, rk: Reranker):
        assert rk.route(rk.threshold_b - 0.01, "faq") == "chat"

    def test_knowledge_never_direct_even_high_score(self, rk: Reranker):
        # 核心不变量：知识库来源无论多高分都不走 faq_direct
        assert rk.route(0.99, "knowledge") == "rag_chat"

    def test_knowledge_boundary_b_goes_rag_chat(self, rk: Reranker):
        # >= threshold_b 含等号
        assert rk.route(rk.threshold_b, "knowledge") == "rag_chat"

    def test_knowledge_low_score_goes_chat(self, rk: Reranker):
        assert rk.route(0.1, "knowledge") == "chat"

    def test_default_source_is_knowledge(self, rk: Reranker):
        assert rk.route(0.99) == "rag_chat"


class TestRerankFallback:
    async def test_empty_documents(self, rk: Reranker):
        assert await rk.rerank("query", []) == []

    async def test_no_api_key_falls_back_to_raw_score(self, rk: Reranker):
        rk._api_key = ""
        docs = [
            {"text": "a", "score": 0.3},
            {"text": "b", "score": 0.9},
            {"text": "c", "score": 0.6},
        ]
        result = await rk.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert all(d["rerank_score"] == d["score"] for d in result)

    async def test_api_failure_falls_back_without_raising(self, rk: Reranker, monkeypatch):
        async def boom(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(rk, "_api_rerank", boom)
        docs = [{"text": "a", "score": 0.5}]
        result = await rk.rerank("query", docs)
        assert result[0]["rerank_score"] == 0.5
