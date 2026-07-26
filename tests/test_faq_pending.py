"""FAQ pending 人工确认流程测试

验证项 3 核心改动：
- 自动命中达阈值后进入 pending（不写 Milvus，不会被 faq_direct 命中）
- approve 后才写入 Milvus 生效；重复 approve 幂等
- reject 后置 demoted，始终不进 Milvus

依赖 PG + Milvus 运行中；使用独立 skill_id=pytest_tmp，不污染生产数据。
"""
from __future__ import annotations

import pytest

from app.config.settings import settings
from app.infrastructure.database import DatabasePool
from app.rag.embedding import embedding_service
from app.rag.faq_promotion import faq_promotion_service
from app.rag.milvus_client import milvus_manager

SKILL = "pytest_tmp"
Q1 = "pytest临时问题一在哪里"
A1 = "pytest临时回答一"
Q2 = "pytest临时问题二怎么办"
A2 = "pytest临时回答二"


async def _cleanup() -> None:
    await DatabasePool.execute("DELETE FROM faq_candidate WHERE skill_id = $1", SKILL)
    client = milvus_manager.get_client()
    if client.has_collection(f"faq_{SKILL}"):
        client.drop_collection(f"faq_{SKILL}")


@pytest.fixture
async def clean_env():
    await faq_promotion_service.ensure_table()
    await _cleanup()
    yield
    await _cleanup()
    # asyncpg 连接池绑定创建时的事件循环，pytest-asyncio 每个测试换新 loop，
    # 必须关闭池让下个测试重建，否则 RuntimeError: Event loop is closed
    await DatabasePool.close()


async def _status(question: str) -> str | None:
    row = await DatabasePool.fetchrow(
        "SELECT status FROM faq_candidate WHERE skill_id = $1 AND question_text = $2",
        SKILL, question,
    )
    return row["status"] if row else None


async def _in_milvus(question: str) -> bool:
    """该问题是否已写入 faq_pytest_tmp 集合（同文本相似度 ~1.0）"""
    client = milvus_manager.get_client()
    coll = f"faq_{SKILL}"
    if not client.has_collection(coll):
        return False
    vec = await embedding_service.embed(question)
    hits = client.search(coll, data=[vec], limit=1, output_fields=["text"])
    return bool(hits and hits[0] and hits[0][0].get("distance", 0.0) > 0.99)


async def _hit_n_times(question: str, answer: str, n: int) -> None:
    for _ in range(n):
        await faq_promotion_service.record_candidate(
            question=question, answer=answer, skill_ids=[SKILL], session_id="pytest-session",
        )


async def test_auto_promotion_goes_pending_not_milvus(clean_env):
    """达阈值 → pending，且绝不写 Milvus（faq_direct 无法命中未确认答案）"""
    threshold = settings.rag.faq_promotion_threshold

    await _hit_n_times(Q1, A1, threshold - 1)
    assert await _status(Q1) == "candidate"

    await _hit_n_times(Q1, A1, 1)  # 第 threshold 次命中
    assert await _status(Q1) == "pending"
    assert not await _in_milvus(Q1), "pending FAQ 不应写入 Milvus"

    # 继续命中也不改变状态
    await _hit_n_times(Q1, A1, 2)
    assert await _status(Q1) == "pending"


async def test_approve_writes_milvus_and_idempotent(clean_env):
    """approve → promoted + 写入 Milvus；重复 approve 幂等"""
    await _hit_n_times(Q1, A1, settings.rag.faq_promotion_threshold)
    assert await _status(Q1) == "pending"

    row = await DatabasePool.fetchrow(
        "SELECT id FROM faq_candidate WHERE skill_id = $1 AND question_text = $2", SKILL, Q1)
    result = await faq_promotion_service.approve(row["id"])
    assert result["status"] == "promoted"
    assert await _status(Q1) == "promoted"
    assert await _in_milvus(Q1), "approve 后应写入 Milvus"

    # 幂等
    again = await faq_promotion_service.approve(row["id"])
    assert again["status"] == "already_promoted"


async def test_reject_never_reaches_milvus(clean_env):
    """reject → demoted，Milvus 中始终没有该 FAQ"""
    await _hit_n_times(Q2, A2, settings.rag.faq_promotion_threshold)
    assert await _status(Q2) == "pending"

    row = await DatabasePool.fetchrow(
        "SELECT id FROM faq_candidate WHERE skill_id = $1 AND question_text = $2", SKILL, Q2)
    result = await faq_promotion_service.reject(row["id"], reason="pytest 驳回")
    assert result["status"] == "rejected"
    assert await _status(Q2) == "demoted"
    assert not await _in_milvus(Q2)


async def test_manual_promote_skips_pending(clean_env):
    """受控数据导入（manual_promote）仍直接 promoted"""
    result = await faq_promotion_service.manual_promote(SKILL, Q2, A2)
    assert result["status"] == "promoted"
    assert await _status(Q2) == "promoted"
    assert await _in_milvus(Q2)
