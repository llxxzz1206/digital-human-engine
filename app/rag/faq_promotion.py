from __future__ import annotations

import hashlib
import logging
from datetime import datetime

from app.config.settings import settings
from app.infrastructure.database import DatabasePool
from app.rag.milvus_client import milvus_manager
from app.rag.embedding import embedding_service

logger = logging.getLogger(__name__)


class FAQPromotionService:
    """FAQ 自动晋升服务

    状态机: candidate → pending → promoted / demoted
    1. rag_chat 交互后，记录用户问题+LLM回答为 FAQ 候选（candidate）
    2. 精确匹配（问题哈希）或相似匹配（Milvus 搜索）累加 hit_count
    3. hit_count 达到阈值 → 置为 pending（待人工确认，暂不写 Milvus，
       因此不会被 faq_direct 命中，避免错误答案被自动放大）
    4. 管理后台 approve() 确认后才写入 Milvus faq_* 集合（promoted）；
       reject() 驳回则置为 demoted
    5. manual_promote()（受控数据导入用）跳过 pending 直接 promoted
    """

    async def ensure_table(self) -> None:
        """确保 FAQ 候选表存在"""
        await DatabasePool.execute("""
            CREATE TABLE IF NOT EXISTS faq_candidate (
                id              SERIAL PRIMARY KEY,
                skill_id        VARCHAR(100) NOT NULL,
                question_hash   VARCHAR(64) NOT NULL,
                question_text   TEXT NOT NULL,
                answer_text     TEXT NOT NULL,
                source_chunks   JSONB DEFAULT '[]',
                hit_count       INTEGER DEFAULT 1,
                last_hit_at     TIMESTAMP DEFAULT NOW(),
                status          VARCHAR(20) DEFAULT 'candidate',
                promoted_at     TIMESTAMP,
                created_at      TIMESTAMP DEFAULT NOW(),
                updated_at      TIMESTAMP DEFAULT NOW(),
                UNIQUE(skill_id, question_hash)
            )
        """)
        # 创建索引
        await DatabasePool.execute("""
            CREATE INDEX IF NOT EXISTS idx_faq_candidate_status
            ON faq_candidate(status)
        """)
        await DatabasePool.execute("""
            CREATE INDEX IF NOT EXISTS idx_faq_candidate_skill_status
            ON faq_candidate(skill_id, status)
        """)

    async def record_candidate(
        self,
        question: str,
        answer: str,
        skill_ids: list[str],
        source_chunks: list[str] | None = None,
        rerank_score: float = 0.0,
        session_id: str = "",
    ) -> None:
        """记录 FAQ 候选（rag_chat 交互后调用）

        流程：
        1. 精确匹配：同一 skill + 相同问题哈希 → hit_count+1
        2. 相似匹配：在 faq_* Milvus 集合中搜索，相似度 >= 阈值 → hit_count+1
        3. 无匹配：创建新 candidate
        4. 检查是否达到晋升阈值 → 置为 pending（待人工确认）
        """
        if not question or not answer:
            return

        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        source_json = __import__("json").dumps(source_chunks or [], ensure_ascii=False)

        for skill_id in skill_ids or []:
            try:
                await self._record_for_skill(
                    skill_id=skill_id,
                    question=question,
                    answer=answer,
                    question_hash=question_hash,
                    source_json=source_json,
                    rerank_score=rerank_score,
                    session_id=session_id,
                )
            except Exception as e:
                logger.error("记录 FAQ 候选失败: skill_id=%s, error=%s", skill_id, e)

    async def _record_for_skill(
        self,
        skill_id: str,
        question: str,
        answer: str,
        question_hash: str,
        source_json: str,
        rerank_score: float,
        session_id: str = "",
    ) -> None:
        """为单个 skill 记录 FAQ 候选"""
        # 1. 精确匹配：同一 skill + 相同问题哈希
        existing = await DatabasePool.fetchrow(
            "SELECT id, hit_count, status, answer_text FROM faq_candidate "
            "WHERE skill_id = $1 AND question_hash = $2",
            skill_id, question_hash,
        )

        if existing:
            new_count = existing["hit_count"] + 1
            # 更新 hit_count 和最后命中时间
            await DatabasePool.execute(
                "UPDATE faq_candidate SET hit_count = $1, last_hit_at = NOW(), "
                "updated_at = NOW(), answer_text = $2 "
                "WHERE id = $3",
                new_count, answer, existing["id"],
            )
            logger.info("FAQ 候选命中: skill_id=%s, hit_count=%d, session=%s, question=%s",
                        skill_id, new_count, session_id or "-", question[:30])

            # 检查是否达到晋升阈值 → 置为 pending（待人工确认，不直接生效）
            if new_count >= settings.rag.faq_promotion_threshold and existing["status"] == "candidate":
                await self._mark_pending(existing["id"], skill_id, question, session_id)
            return

        # 2. 相似匹配：在 faq_* Milvus 集合中搜索
        similar_candidate = await self._find_similar_candidate(question, skill_id)
        if similar_candidate:
            new_count = similar_candidate["hit_count"] + 1
            await DatabasePool.execute(
                "UPDATE faq_candidate SET hit_count = $1, last_hit_at = NOW(), "
                "updated_at = NOW() WHERE id = $2",
                new_count, similar_candidate["id"],
            )
            logger.info("FAQ 相似候选命中: skill_id=%s, hit_count=%d, session=%s, question=%s",
                        skill_id, new_count, session_id or "-", question[:30])

            if new_count >= settings.rag.faq_promotion_threshold and similar_candidate["status"] == "candidate":
                await self._mark_pending(similar_candidate["id"], skill_id,
                                         similar_candidate["question_text"], session_id)
            return

        # 3. 新建候选
        await DatabasePool.execute(
            """INSERT INTO faq_candidate
               (skill_id, question_hash, question_text, answer_text, source_chunks, hit_count, status)
               VALUES ($1, $2, $3, $4, $5::jsonb, 1, 'candidate')""",
            skill_id, question_hash, question, answer, source_json,
        )
        logger.info("FAQ 新候选: skill_id=%s, question=%s", skill_id, question[:30])

    async def _find_similar_candidate(self, question: str, skill_id: str) -> dict | None:
        """在 Milvus faq_* 集合中搜索相似问题，返回对应的 PG 候选记录"""
        collection_name = f"faq_{skill_id}"
        client = milvus_manager.get_client()

        try:
            if not client.has_collection(collection_name):
                return None

            # 向量化查询
            query_vector = await embedding_service.embed(question)

            hits = client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=1,
                output_fields=["text", "metadata"],
            )

            if not hits or not hits[0]:
                return None

            top_hit = hits[0][0]
            similarity = top_hit.get("distance", 0.0)

            if similarity < settings.rag.faq_similarity_threshold:
                return None

            # 通过 metadata 中的 source_candidate_id 查找 PG 记录
            metadata = top_hit.get("entity", {}).get("metadata", {})
            candidate_id = metadata.get("source_candidate_id")

            if not candidate_id:
                return None

            row = await DatabasePool.fetchrow(
                "SELECT id, hit_count, status, question_text, answer_text FROM faq_candidate "
                "WHERE id = $1 AND status = 'candidate'",
                candidate_id,
            )
            return dict(row) if row else None

        except Exception as e:
            logger.error("FAQ 相似匹配异常: skill_id=%s, error=%s", skill_id, e)
            return None

    async def _mark_pending(self, candidate_id: int, skill_id: str, question: str,
                            session_id: str = "") -> None:
        """达到命中阈值 → 置为 pending（待人工确认）

        只更新 PG 状态，不写 Milvus：pending FAQ 不会被 faq_direct 命中，
        避免 LLM 连续答错时错误答案被自动固化为"标准答案"。
        """
        try:
            await DatabasePool.execute(
                "UPDATE faq_candidate SET status = 'pending', updated_at = NOW() WHERE id = $1",
                candidate_id,
            )
            logger.info("FAQ 进入待审核: skill_id=%s, candidate_id=%d, session=%s, question=%s",
                        skill_id, candidate_id, session_id or "-", question[:30])
        except Exception as e:
            logger.error("FAQ 置 pending 失败: candidate_id=%d, error=%s", candidate_id, e)

    async def approve(self, candidate_id: int) -> dict:
        """人工确认通过：pending/candidate → promoted，写入 Milvus 正式生效"""
        row = await DatabasePool.fetchrow(
            "SELECT skill_id, question_text, answer_text, status FROM faq_candidate WHERE id = $1",
            candidate_id,
        )
        if not row:
            return {"status": "not_found"}
        if row["status"] == "promoted":
            return {"status": "already_promoted", "id": candidate_id}

        await self._promote(candidate_id, row["skill_id"], row["question_text"], row["answer_text"])

        # _promote 内部捕获异常，需复核状态确认是否真的写入成功
        after = await DatabasePool.fetchrow(
            "SELECT status FROM faq_candidate WHERE id = $1", candidate_id
        )
        if after and after["status"] == "promoted":
            logger.info("FAQ 人工确认通过: candidate_id=%d, skill_id=%s", candidate_id, row["skill_id"])
            return {"status": "promoted", "id": candidate_id}
        return {"status": "promote_failed", "id": candidate_id}

    async def reject(self, candidate_id: int, reason: str = "") -> dict:
        """人工驳回：pending/candidate → demoted（从未写入 Milvus，无需删除向量）"""
        row = await DatabasePool.fetchrow(
            "SELECT status FROM faq_candidate WHERE id = $1", candidate_id
        )
        if not row:
            return {"status": "not_found"}
        if row["status"] == "promoted":
            return {"status": "already_promoted_use_demote"}

        await DatabasePool.execute(
            "UPDATE faq_candidate SET status = 'demoted', updated_at = NOW() WHERE id = $1",
            candidate_id,
        )
        logger.info("FAQ 人工驳回: candidate_id=%d, reason=%s", candidate_id, reason)
        return {"status": "rejected", "id": candidate_id}

    async def _promote(self, candidate_id: int, skill_id: str, question: str, answer: str) -> None:
        """将候选晋升为 FAQ，写入 Milvus faq_* 集合"""
        try:
            # 1. 向量化问题
            vector = await embedding_service.embed(question)

            # 2. 确保 faq_* 集合存在
            collection_name = f"faq_{skill_id}"
            milvus_manager.ensure_collection(collection_name)

            # 3. 写入 Milvus
            client = milvus_manager.get_client()
            client.insert(
                collection_name=collection_name,
                data=[{
                    "vector": vector,
                    "text": answer,
                    "metadata": {
                        "question": question,
                        "skill_id": skill_id,
                        "source_candidate_id": candidate_id,
                        "promoted_at": datetime.now().isoformat(),
                    },
                }],
            )

            # 4. 更新 PG 状态
            await DatabasePool.execute(
                "UPDATE faq_candidate SET status = 'promoted', promoted_at = NOW(), "
                "updated_at = NOW() WHERE id = $1",
                candidate_id,
            )

            logger.info("FAQ 晋升成功: skill_id=%s, question=%s, candidate_id=%d",
                        skill_id, question[:30], candidate_id)

        except Exception as e:
            logger.error("FAQ 晋升失败: candidate_id=%d, error=%s", candidate_id, e)

    async def manual_promote(self, skill_id: str, question: str, answer: str) -> dict:
        """手动晋升：直接创建 FAQ 条目（跳过 hit_count 判断）"""
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()

        # 检查是否已存在
        existing = await DatabasePool.fetchrow(
            "SELECT id, status, answer_text FROM faq_candidate "
            "WHERE skill_id = $1 AND question_hash = $2",
            skill_id, question_hash,
        )

        if existing and existing["status"] == "promoted":
            return {"status": "already_promoted", "id": existing["id"]}

        # 如果已有 candidate，更新 answer 并直接晋升
        if existing:
            # 使用 API 传入的 answer 替代旧的
            await DatabasePool.execute(
                "UPDATE faq_candidate SET answer_text = $1, updated_at = NOW() WHERE id = $2",
                answer, existing["id"],
            )
            await self._promote(existing["id"], skill_id, question, answer)
            return {"status": "promoted", "id": existing["id"]}

        # 新建并直接晋升
        await DatabasePool.execute(
            """INSERT INTO faq_candidate
               (skill_id, question_hash, question_text, answer_text, source_chunks, hit_count, status)
               VALUES ($1, $2, $3, $4, '[]'::jsonb, 0, 'candidate')""",
            skill_id, question_hash, question, answer,
        )

        row = await DatabasePool.fetchrow(
            "SELECT id FROM faq_candidate WHERE skill_id = $1 AND question_hash = $2",
            skill_id, question_hash,
        )

        await self._promote(row["id"], skill_id, question, answer)
        return {"status": "promoted", "id": row["id"]}

    async def demote(self, candidate_id: int, reason: str = "") -> dict:
        """降级 FAQ（从 Milvus 删除 + PG 状态更新）"""
        candidate = await DatabasePool.fetchrow(
            "SELECT skill_id, question_text, status FROM faq_candidate WHERE id = $1",
            candidate_id,
        )

        if not candidate:
            return {"status": "not_found"}

        if candidate["status"] != "promoted":
            return {"status": "not_promoted"}

        skill_id = candidate["skill_id"]

        # 从 Milvus 删除
        try:
            collection_name = f"faq_{skill_id}"
            client = milvus_manager.get_client()
            if client.has_collection(collection_name):
                client.delete(
                    collection_name=collection_name,
                    filter=f'metadata["source_candidate_id"] == {candidate_id}',
                )
        except Exception as e:
            logger.error("FAQ Milvus 删除失败: %s", e)

        # 更新 PG 状态
        await DatabasePool.execute(
            "UPDATE faq_candidate SET status = 'demoted', updated_at = NOW() WHERE id = $1",
            candidate_id,
        )

        logger.info("FAQ 降级: candidate_id=%d, skill_id=%s, reason=%s",
                    candidate_id, skill_id, reason)
        return {"status": "demoted", "id": candidate_id}

    async def list_candidates(
        self,
        skill_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """列出 FAQ 候选"""
        conditions = []
        params = []
        idx = 1

        if skill_id:
            conditions.append(f"skill_id = ${idx}")
            params.append(skill_id)
            idx += 1
        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await DatabasePool.fetch(
            f"SELECT id, skill_id, question_text, answer_text, hit_count, status, "
            f"promoted_at, created_at, updated_at FROM faq_candidate "
            f"{where} ORDER BY updated_at DESC LIMIT ${idx}",
            *params,
        )
        return [dict(r) for r in rows]

    async def get_stats(self, skill_id: str | None = None) -> dict:
        """获取 FAQ 统计信息"""
        condition = "WHERE skill_id = $1" if skill_id else ""
        params = [skill_id] if skill_id else []

        rows = await DatabasePool.fetch(
            f"SELECT status, COUNT(*) as count FROM faq_candidate "
            f"{condition} GROUP BY status",
            *params,
        )

        stats = {r["status"]: r["count"] for r in rows}
        return {
            "candidate": stats.get("candidate", 0),
            "pending": stats.get("pending", 0),
            "promoted": stats.get("promoted", 0),
            "demoted": stats.get("demoted", 0),
        }

    async def batch_import(self, skill_id: str, items: list[dict]) -> dict:
        """批量导入 FAQ（初始化场景）"""
        promoted = 0
        for item in items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            if question and answer:
                result = await self.manual_promote(skill_id, question, answer)
                if result["status"] == "promoted":
                    promoted += 1

        return {"total": len(items), "promoted": promoted}


faq_promotion_service = FAQPromotionService()
