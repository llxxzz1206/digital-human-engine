from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config.settings import settings
from app.infrastructure.redis import RedisPool
from app.session.models import Session

logger = logging.getLogger(__name__)

# Key 前缀，与 Java 端 sessionKeyPrefix 对齐
_KEY_PREFIX = "digitalhuman:session:"


class SessionManager:
    """基于 Redis 的会话管理器，与 Java 端共享同一数据源"""

    def __init__(self) -> None:
        # 内存回退（Redis 不可用时使用）
        self._fallback: dict[str, dict[str, Any]] = {}
        self._fallback_lock = asyncio.Lock()

    async def _get_redis(self):
        """获取 Redis 客户端，不可用则返回 None"""
        try:
            redis = await RedisPool.get()
            if await redis.ping():
                return redis
        except Exception:
            pass
        return None

    async def create_session(
        self,
        session_id: str,
        user_id: str = "",
        avatar_id: str = "",
        scene_id: str = "",
        device_id: str = "",
        device_location: str = "",
        user_type: str = "general",
        platform: str = "fixed_terminal",
        current_location: str = "",
        location_source: str = "",
        capabilities: list[str] | None = None,
    ) -> Session:
        """创建会话

        Args:
            user_id: 用户ID（移动端必填）
            avatar_id: 形象ID
            scene_id: 场景ID
            device_id: 设备ID（固定终端必填）
            device_location: 设备位置（固定终端）
            user_type: 用户类型（影响讲解风格）
            platform: 平台类型（fixed_terminal/mobile_app/mini_app）
            current_location: 实时位置（移动端，GPS推断）
            location_source: 位置来源（gps/beacon/manual）
        """
        now = int(time.time() * 1000)
        session = Session(
            sessionId=session_id,
            userId=user_id,
            avatarId=avatar_id,
            mountedSkills=[],
            platform=platform,
            sceneId=scene_id,
            deviceId=device_id,
            deviceLocation=device_location,
            currentLocation=current_location,
            locationSource=location_source,
            locationUpdatedAt=now if current_location else 0,
            userType=user_type,
            capabilities=capabilities or ["text", "audio", "avatar"],
            createdAt=now,
            updatedAt=now,
        )
        key = f"{_KEY_PREFIX}{session_id}"
        data = session.model_dump_json()

        redis = await self._get_redis()
        if redis is not None:
            await redis.set(key, data, ex=settings.redis.session_ttl)
            logger.info("会话已创建(Redis): %s", session_id)
        else:
            async with self._fallback_lock:
                self._fallback[session_id] = session.model_dump()
            logger.info("会话已创建(内存回退): %s", session_id)

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """获取会话"""
        key = f"{_KEY_PREFIX}{session_id}"

        redis = await self._get_redis()
        if redis is not None:
            data = await redis.get(key)
            if data is None:
                return None
            return Session.model_validate_json(data)
        else:
            async with self._fallback_lock:
                raw = self._fallback.get(session_id)
            if raw is None:
                return None
            return Session.model_validate(raw)

    async def destroy_session(self, session_id: str) -> None:
        """销毁会话"""
        key = f"{_KEY_PREFIX}{session_id}"

        redis = await self._get_redis()
        if redis is not None:
            await redis.delete(key)
            logger.info("会话已销毁(Redis): %s", session_id)
        else:
            async with self._fallback_lock:
                self._fallback.pop(session_id, None)
            logger.info("会话已销毁(内存回退): %s", session_id)

    async def mount_skill(self, session_id: str, skill_id: str) -> None:
        """挂载 Skill 到会话"""
        session = await self.get_session(session_id)
        if session is None:
            logger.error("会话不存在: %s", session_id)
            return
        if skill_id not in session.mountedSkills:
            session.mountedSkills.append(skill_id)
            session.updatedAt = int(time.time() * 1000)
            await self.save_session(session)
            logger.info("Skill 已挂载: session=%s, skill=%s", session_id, skill_id)

    async def unmount_skill(self, session_id: str, skill_id: str) -> None:
        """卸载会话中的 Skill"""
        session = await self.get_session(session_id)
        if session is None:
            logger.error("会话不存在: %s", session_id)
            return
        if skill_id in session.mountedSkills:
            session.mountedSkills.remove(skill_id)
            session.updatedAt = int(time.time() * 1000)
            await self.save_session(session)
            logger.info("Skill 已卸载: session=%s, skill=%s", session_id, skill_id)

    async def refresh_ttl(self, session_id: str) -> None:
        """刷新会话 TTL"""
        key = f"{_KEY_PREFIX}{session_id}"

        redis = await self._get_redis()
        if redis is not None:
            await redis.expire(key, settings.redis.session_ttl)

    async def save_session(self, session: Session) -> None:
        """保存会话到存储"""
        key = f"{_KEY_PREFIX}{session.sessionId}"
        data = session.model_dump_json()

        redis = await self._get_redis()
        if redis is not None:
            await redis.set(key, data, ex=settings.redis.session_ttl)
        else:
            async with self._fallback_lock:
                self._fallback[session.sessionId] = session.model_dump()


session_manager = SessionManager()
