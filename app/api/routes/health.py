from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.infrastructure.redis import RedisPool

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    redis_ok = await RedisPool.ping()
    return {
        "status": "ok" if redis_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "redis": "connected" if redis_ok else "disconnected",
    }
