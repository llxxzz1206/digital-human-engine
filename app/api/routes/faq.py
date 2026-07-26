from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from app.rag.faq_promotion import faq_promotion_service

logger = logging.getLogger(__name__)

# 注意：vite/nginx 代理会剥掉 /api 前缀，后端路由约定不带 /api（否则代理后 404）
router = APIRouter(prefix="/faq", tags=["faq"])


# ── Request Models ─────────────────────────────────────────


class PromoteRequest(BaseModel):
    skill_id: str
    question: str
    answer: str


class DemoteRequest(BaseModel):
    candidate_id: int
    reason: str = ""


class ApproveRequest(BaseModel):
    candidate_id: int


class RejectRequest(BaseModel):
    candidate_id: int
    reason: str = ""


class ImportRequest(BaseModel):
    skill_id: str
    items: list[dict]  # [{"question": "...", "answer": "..."}]


# ── Endpoints ──────────────────────────────────────────────


@router.get("/candidates")
async def list_candidates(
    skill_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
):
    """列出 FAQ 候选"""
    try:
        candidates = await faq_promotion_service.list_candidates(skill_id, status, limit)
        return {"candidates": candidates}
    except Exception as e:
        logger.error("列出 FAQ 候选失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_faqs(
    skill_id: str | None = None,
    limit: int = 50,
):
    """列出已晋升的 FAQ"""
    try:
        candidates = await faq_promotion_service.list_candidates(skill_id, "promoted", limit)
        return {"faqs": candidates}
    except Exception as e:
        logger.error("列出 FAQ 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote")
async def manual_promote(req: PromoteRequest):
    """手动晋升 FAQ"""
    try:
        result = await faq_promotion_service.manual_promote(req.skill_id, req.question, req.answer)
        return result
    except Exception as e:
        logger.error("手动晋升 FAQ 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demote")
async def demote(req: DemoteRequest):
    """降级 FAQ"""
    try:
        result = await faq_promotion_service.demote(req.candidate_id, req.reason)
        return result
    except Exception as e:
        logger.error("降级 FAQ 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve")
async def approve(req: ApproveRequest):
    """人工确认通过：pending FAQ 正式写入向量库生效"""
    try:
        result = await faq_promotion_service.approve(req.candidate_id)
        return result
    except Exception as e:
        logger.error("确认 FAQ 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reject")
async def reject(req: RejectRequest):
    """人工驳回：pending FAQ 置为 demoted，不写入向量库"""
    try:
        result = await faq_promotion_service.reject(req.candidate_id, req.reason)
        return result
    except Exception as e:
        logger.error("驳回 FAQ 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(skill_id: str | None = None):
    """获取 FAQ 统计信息"""
    try:
        stats = await faq_promotion_service.get_stats(skill_id)
        return stats
    except Exception as e:
        logger.error("获取 FAQ 统计失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def batch_import(req: ImportRequest):
    """批量导入 FAQ（初始化场景）"""
    try:
        result = await faq_promotion_service.batch_import(req.skill_id, req.items)
        return result
    except Exception as e:
        logger.error("批量导入 FAQ 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
