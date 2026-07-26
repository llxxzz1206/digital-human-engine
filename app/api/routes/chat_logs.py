from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.chat_logger import chat_logger

# 注意：vite/nginx 代理会剥掉 /api 前缀，后端路由约定不带 /api（否则代理后 404）
router = APIRouter(tags=["chat-logs"])


@router.get("/chat-logs/dates")
async def list_dates() -> dict:
    """返回有对话日志的日期列表"""
    dates = chat_logger.list_dates()
    return {"dates": dates}


@router.get("/chat-logs")
async def get_chat_logs(
    date: str | None = Query(None, description="日期，如 2026-07-18，默认今天"),
    session_id: str | None = Query(None, description="按会话ID过滤"),
) -> dict:
    """获取对话日志"""
    records = chat_logger.read_logs(date=date, session_id=session_id)
    return {"records": records, "count": len(records)}
