from __future__ import annotations

import logging

from app.rag.milvus_client import milvus_manager
from app.rag.embedding import embedding_service
from app.rag.reranker import reranker
from app.config.settings import settings

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG 向量检索器，基于 Milvus skill_* 集合"""

    async def search(
        self,
        query: str,
        skill_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """向量检索

        Args:
            query: 查询文本
            skill_ids: 限定搜索范围的 Skill ID 列表
            top_k: 返回前 K 个结果（默认使用配置值）

        Returns:
            检索结果列表，每个结果包含 {"text": str, "metadata": dict, "score": float, "source_type": str}
        """
        if top_k is None:
            top_k = settings.rag.top_k
        
        if not skill_ids:
            skill_ids = self._discover_skill_collections()
            if not skill_ids:
                logger.info("RAG 检索: 无 skill_ids 且无可用 collection，跳过检索")
                return []
            logger.info("RAG 检索: skill_ids 为空，自动发现 collections: %s", skill_ids)

        query_vector = await embedding_service.embed(query)

        all_results: list[dict] = []
        client = milvus_manager.get_client()

        for skill_id in skill_ids:
            collection_name = f"skill_{skill_id}"

            try:
                if not client.has_collection(collection_name):
                    logger.debug("Collection 不存在: %s", collection_name)
                    continue

                hits = client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "metadata"],
                )

                for hit_list in hits:
                    for hit in hit_list:
                        all_results.append({
                            "text": hit.get("entity", {}).get("text", ""),
                            "metadata": hit.get("entity", {}).get("metadata", {}),
                            "score": hit.get("distance", 0.0),
                            "skill_id": skill_id,
                            "source_type": "knowledge",
                        })
            except Exception as e:
                logger.error("RAG 检索异常: collection=%s, error=%s", collection_name, e)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        results = all_results[:top_k]

        logger.info("RAG 检索完成: query=%s, skill_ids=%s, results=%d", query[:30], skill_ids, len(results))
        return results

    def _discover_skill_collections(self) -> list[str]:
        """自动发现 Milvus 中所有 skill_* collection，返回 skill_id 列表"""
        try:
            client = milvus_manager.get_client()
            collections = client.list_collections()
            skill_ids = [c.replace("skill_", "", 1) for c in collections if c.startswith("skill_")]
            return skill_ids
        except Exception as e:
            logger.error("发现 skill collections 失败: %s", e)
            return []


class FAQRetriever:
    """FAQ 向量检索器，基于 Milvus faq_* 集合

    faq_* 集合中 text 字段存的是 LLM 润色后的回答，
    vector 字段存的是问题的 embedding（用于匹配用户查询）。
    """

    async def search(
        self,
        query: str,
        skill_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """FAQ 向量检索"""
        if top_k is None:
            top_k = settings.rag.top_k
            
        if not skill_ids:
            skill_ids = self._discover_faq_collections()
            if not skill_ids:
                return []

        query_vector = await embedding_service.embed(query)

        all_results: list[dict] = []
        client = milvus_manager.get_client()

        for skill_id in skill_ids:
            collection_name = f"faq_{skill_id}"

            try:
                if not client.has_collection(collection_name):
                    continue

                hits = client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "metadata"],
                )

                for hit_list in hits:
                    for hit in hit_list:
                        all_results.append({
                            "text": hit.get("entity", {}).get("text", ""),
                            "metadata": hit.get("entity", {}).get("metadata", {}),
                            "score": hit.get("distance", 0.0),
                            "skill_id": skill_id,
                            "source_type": "faq",
                        })
            except Exception as e:
                logger.error("FAQ 检索异常: collection=%s, error=%s", collection_name, e)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        results = all_results[:top_k]

        logger.info("FAQ 检索完成: query=%s, skill_ids=%s, results=%d", query[:30], skill_ids, len(results))
        return results

    def _discover_faq_collections(self) -> list[str]:
        """自动发现 Milvus 中所有 faq_* collection，返回 skill_id 列表"""
        try:
            client = milvus_manager.get_client()
            collections = client.list_collections()
            skill_ids = [c.replace("faq_", "", 1) for c in collections if c.startswith("faq_")]
            return skill_ids
        except Exception as e:
            logger.error("发现 faq collections 失败: %s", e)
            return []


class DualCollectionRetriever:
    """双集合检索器：先查 FAQ，再查知识库

    支持多场景多设备的分层检索：
    - 有 scene_id/device_id 时：设备级 FAQ → 场景级 FAQ → 设备级知识库 → 场景级知识库
    - 无 scene_id/device_id 时：自动发现所有 collection（向后兼容）

    路由规则：
    - FAQ 高分命中 → faq_direct（零 LLM 调用）
    - 知识库命中 → rag_chat（LLM 生成自然回答）
    - 均低分 → chat（普通对话）
    """

    def __init__(self) -> None:
        self._faq_retriever = FAQRetriever()
        self._knowledge_retriever = RAGRetriever()

    async def search(
        self,
        query: str,
        skill_ids: list[str] | None = None,
        scene_id: str = "",
        device_id: str = "",
        platform: str = "fixed_terminal",
    ) -> tuple[list[dict], str]:
        """双集合检索（支持分层）

        Args:
            query: 查询文本
            skill_ids: 限定搜索范围的 Skill ID 列表
            scene_id: 场景 ID（用于分层检索）
            device_id: 设备 ID（用于分层检索）
            platform: 客户端平台（fixed_terminal/mobile_app/mini_app/web_admin）

        Returns:
            (results, source_type):
                source_type = "faq" | "knowledge"
        """
        if not settings.rag.faq_enabled:
            results = await self._knowledge_retriever.search(query, skill_ids)
            return results, "knowledge"

        # 分层检索：有 scene_id/device_id 时按层级搜索
        if scene_id or device_id:
            return await self._hierarchical_search(query, skill_ids, scene_id, device_id, platform=platform)

        # 兼容模式：无 scene_id/device_id，走原有逻辑
        return await self._flat_search(query, skill_ids)

    async def _hierarchical_search(
        self,
        query: str,
        skill_ids: list[str] | None,
        scene_id: str,
        device_id: str,
        platform: str = "fixed_terminal",
    ) -> tuple[list[dict], str]:
        """分层检索：设备级 → 场景级（支持移动端）

        检索优先级：
        1. 设备级 FAQ (faq_{device_id}) - 仅固定终端
        2. 场景级 FAQ (faq_{scene_id})
        3. 设备级知识库 (skill_{device_id}) - 仅固定终端
        4. 场景级知识库 (skill_{scene_id}) 或指定的 skill_ids
        """
        # ── 固定终端：设备级 FAQ ──
        if platform == "fixed_terminal" and device_id:
            faq_results = await self._faq_retriever.search(query, [device_id], top_k=3)
            if faq_results:
                faq_results = await reranker.rerank(query, faq_results, top_k=3)
                max_faq_score = max(r.get("rerank_score", 0.0) for r in faq_results) if faq_results else 0.0
                if max_faq_score >= settings.rag.rerank_threshold_b:
                    logger.info("设备级 FAQ 命中: device=%s, score=%.4f", device_id, max_faq_score)
                    return faq_results, "faq"

        # ── 场景级 FAQ（固定终端 + 移动端）──
        if scene_id:
            faq_results = await self._faq_retriever.search(query, [scene_id], top_k=3)
            if faq_results:
                faq_results = await reranker.rerank(query, faq_results, top_k=3)
                max_faq_score = max(r.get("rerank_score", 0.0) for r in faq_results) if faq_results else 0.0
                if max_faq_score >= settings.rag.rerank_threshold_b:
                    logger.info("场景级 FAQ 命中: scene=%s, score=%.4f", scene_id, max_faq_score)
                    return faq_results, "faq"

        # ── 固定终端：设备级知识库 ──
        if platform == "fixed_terminal" and device_id:
            knowledge_results = await self._knowledge_retriever.search(query, [device_id])
            if knowledge_results:
                logger.info("设备级知识库命中: device=%s, results=%d", device_id, len(knowledge_results))
                return knowledge_results, "knowledge"

        # ── 场景级知识库 / 指定 skill_ids（固定终端 + 移动端）──
        search_ids = skill_ids if skill_ids else ([scene_id] if scene_id else None)
        knowledge_results = await self._knowledge_retriever.search(query, search_ids)
        return knowledge_results, "knowledge"

    async def _flat_search(
        self,
        query: str,
        skill_ids: list[str] | None,
    ) -> tuple[list[dict], str]:
        """扁平检索：无场景/设备信息时的原有逻辑"""
        # 第一步：检索 FAQ
        faq_results = await self._faq_retriever.search(query, skill_ids, top_k=3)

        if faq_results:
            faq_results = await reranker.rerank(query, faq_results, top_k=3)
            max_faq_score = max(r.get("rerank_score", 0.0) for r in faq_results) if faq_results else 0.0

            if max_faq_score >= settings.rag.rerank_threshold_b:
                logger.info("FAQ 命中: query=%s, max_score=%.4f", query[:30], max_faq_score)
                return faq_results, "faq"

        # 第二步：FAQ 未高分命中，检索知识库
        knowledge_results = await self._knowledge_retriever.search(query, skill_ids)
        return knowledge_results, "knowledge"


# 保留原有 rag_retriever 实例（向后兼容）
rag_retriever = RAGRetriever()

# 双集合检索器实例
dual_retriever = DualCollectionRetriever()
