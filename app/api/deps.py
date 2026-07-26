"""API 认证依赖"""
from __future__ import annotations

import logging

from fastapi import Header, HTTPException

from app.config.settings import settings

logger = logging.getLogger(__name__)


async def verify_admin_token(x_admin_token: str = Header(default="")):
    """Admin API 鉴权：校验请求头 X-Admin-Token"""
    if not settings.admin_token:
        # 开发环境未配置 token 时放行，但打 warn
        logger.warning("admin_token 未配置，Admin API 处于无鉴权状态（仅限开发环境）")
        return
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")
