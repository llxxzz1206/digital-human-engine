"""用户中心 - 用户认证、会话管理、偏好持久化"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.config.settings import settings
from app.infrastructure.redis import RedisPool
from app.session.models import Session, UserType

logger = logging.getLogger(__name__)

# 用户数据 Key 前缀
_USER_PREFIX = "digitalhuman:user:"
_SESSION_BY_USER_PREFIX = "digitalhuman:user_session:"


class UserPreference:
    """用户偏好设置"""

    def __init__(self) -> None:
        self.user_type: UserType = "general"
        self.frequent_scenes: list[str] = []  # 常用场景
        self.favorite_skills: list[str] = []  # 常用技能
        self.last_active_scene: str = ""      # 上次活跃场景


class UserManager:
    """用户管理器

    功能：
    1. 用户认证（验证 userId）
    2. 用户级会话管理（多设备同步）
    3. 用户偏好持久化
    4. 对话历史跨设备同步
    """

    def __init__(self) -> None:
        self._fallback_user: dict[str, dict[str, Any]] = {}
        self._fallback_session: dict[str, str] = {}

    async def _get_redis(self):
        """获取 Redis 客户端"""
        try:
            redis = await RedisPool.get()
            if await redis.ping():
                return redis
        except Exception:
            pass
        return None

    async def validate_user(self, user_id: str) -> bool:
        """验证用户 ID（简单校验，可扩展为数据库查询）

        Args:
            user_id: 用户 ID

        Returns:
            True = 有效用户，False = 无效用户
        """
        if not user_id or not user_id.strip():
            return False

        # 简单规则：user_ 开头或数字开头
        # 生产环境应查询用户表
        return user_id.startswith("user_") or user_id[0].isdigit()

    async def get_or_create_session(
        self,
        user_id: str,
        device_id: str,
        platform: str = "mobile_app",
        scene_id: str = "",
    ) -> tuple[Session, bool]:
        """获取或创建用户会话（多设备同步）

        Args:
            user_id: 用户 ID
            device_id: 设备 ID
            platform: 平台类型
            scene_id: 初始场景 ID

        Returns:
            (Session, is_new) 元组：
            - Session: 会话对象
            - is_new: 是否为新创建
        """
        # 查找用户活跃会话
        active_session_id = await self._get_active_session(user_id)

        if active_session_id:
            # 恢复现有会话，更新设备绑定
            from app.session.manager import session_manager
            session = await session_manager.get_session(active_session_id)

            if session:
                # 更新设备绑定
                session.deviceId = device_id
                session.updatedAt = int(time.time() * 1000)
                await session_manager.save_session(session)
                logger.info("用户会话恢复: user=%s, session=%s, device=%s", user_id, active_session_id, device_id)
                return session, False

        # 创建新会话
        import uuid
        session_id = str(uuid.uuid4())
        from app.session.manager import session_manager

        session = await session_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            device_id=device_id,
            platform=platform,
            scene_id=scene_id,
        )

        # 记录用户活跃会话
        await self._set_active_session(user_id, session_id)

        logger.info("用户会话创建: user=%s, session=%s, device=%s", user_id, session_id, device_id)
        return session, True

    async def _get_active_session(self, user_id: str) -> str | None:
        """获取用户活跃会话 ID"""
        key = f"{_SESSION_BY_USER_PREFIX}{user_id}"

        redis = await self._get_redis()
        if redis:
            return await redis.get(key)
        else:
            return self._fallback_session.get(user_id)

    async def _set_active_session(self, user_id: str, session_id: str) -> None:
        """设置用户活跃会话"""
        key = f"{_SESSION_BY_USER_PREFIX}{user_id}"

        redis = await self._get_redis()
        if redis:
            # 24 小时过期（比会话 TTL 长）
            await redis.set(key, session_id, ex=86400)
        else:
            self._fallback_session[user_id] = session_id

    async def get_user_preference(self, user_id: str) -> UserPreference:
        """获取用户偏好"""
        key = f"{_USER_PREFIX}{user_id}"

        redis = await self._get_redis()
        if redis:
            data = await redis.hgetall(key)
            if data:
                pref = UserPreference()
                pref.user_type = data.get("user_type", "general")
                pref.frequent_scenes = data.get("frequent_scenes", "").split(",") if data.get("frequent_scenes") else []
                pref.favorite_skills = data.get("favorite_skills", "").split(",") if data.get("favorite_skills") else []
                pref.last_active_scene = data.get("last_active_scene", "")
                return pref

        return UserPreference()

    async def update_user_preference(
        self,
        user_id: str,
        user_type: str | None = None,
        frequent_scenes: list[str] | None = None,
        favorite_skills: list[str] | None = None,
        last_active_scene: str | None = None,
    ) -> None:
        """更新用户偏好"""
        key = f"{_USER_PREFIX}{user_id}"

        mapping = {}
        if user_type:
            mapping["user_type"] = user_type
        if frequent_scenes is not None:
            mapping["frequent_scenes"] = ",".join(frequent_scenes)
        if favorite_skills is not None:
            mapping["favorite_skills"] = ",".join(favorite_skills)
        if last_active_scene:
            mapping["last_active_scene"] = last_active_scene

        if not mapping:
            return

        redis = await self._get_redis()
        if redis:
            await redis.hset(key, mapping=mapping)
        else:
            if user_id not in self._fallback_user:
                self._fallback_user[user_id] = {}
            self._fallback_user[user_id].update(mapping)


# 全局实例
user_manager = UserManager()