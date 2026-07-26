from __future__ import annotations

import json
import logging

from app.config.settings import settings
from app.infrastructure.redis import RedisPool

logger = logging.getLogger(__name__)

# Key 前缀
_KEY_PREFIX = "digitalhuman:history:"
_SUMMARY_KEY_PREFIX = "digitalhuman:summary:"

# 最大保留轮数（一问一答 = 2 条消息）
MAX_MESSAGES = 80  # 40 轮 × 2 条


class ConversationHistory:
    """对话历史管理器，基于 Redis List 存储"""

    def __init__(self) -> None:
        self._fallback: dict[str, list[str]] = {}

    async def _get_redis(self):
        """获取 Redis 客户端（连接失败返回 None）"""
        try:
            return await RedisPool.get()
        except Exception:
            return None

    async def append(self, session_id: str, role: str, content: str | dict) -> None:
        """追加一条消息到对话历史

        Args:
            session_id: 会话 ID
            role: 角色 (system/user/assistant/tool)
            content: 消息内容
        """
        key = f"{_KEY_PREFIX}{session_id}"
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)

        redis = await self._get_redis()
        if redis is not None:
            await redis.rpush(key, msg)
            # 保留最近 N 条消息
            await redis.ltrim(key, -MAX_MESSAGES, -1)
            await redis.expire(key, settings.redis.session_ttl)
        else:
            if session_id not in self._fallback:
                self._fallback[session_id] = []
            self._fallback[session_id].append(msg)
            # 内存模式也裁剪
            if len(self._fallback[session_id]) > MAX_MESSAGES:
                self._fallback[session_id] = self._fallback[session_id][-MAX_MESSAGES:]

    async def get_messages(self, session_id: str) -> list[dict]:
        """获取完整对话历史"""
        key = f"{_KEY_PREFIX}{session_id}"

        redis = await self._get_redis()
        if redis is not None:
            items = await redis.lrange(key, 0, -1)
            return [json.loads(item) for item in items]
        else:
            raw = self._fallback.get(session_id, [])
            return [json.loads(item) for item in raw]

    async def clear(self, session_id: str) -> None:
        """清空对话历史"""
        key = f"{_KEY_PREFIX}{session_id}"

        redis = await self._get_redis()
        if redis is not None:
            await redis.delete(key)
            await redis.delete(f"{_SUMMARY_KEY_PREFIX}{session_id}")
        else:
            self._fallback.pop(session_id, None)

        logger.info("对话历史已清空: %s", session_id)

    # ── 摘要缓存 ──────────────────────────────────────────

    async def get_summary(self, session_id: str) -> tuple[str | None, int]:
        """获取缓存的摘要及生成时的消息数

        Returns:
            (summary_text, message_count_at_generation) 或 (None, 0)
        """
        key = f"{_SUMMARY_KEY_PREFIX}{session_id}"
        redis = await self._get_redis()
        if redis is not None:
            raw = await redis.get(key)
            if raw:
                data = json.loads(raw)
                return data.get("summary"), data.get("msg_count", 0)
        return None, 0

    async def set_summary(self, session_id: str, summary: str, msg_count: int) -> None:
        """缓存摘要，附带生成时的消息数（用于判断是否需要刷新）"""
        key = f"{_SUMMARY_KEY_PREFIX}{session_id}"
        payload = json.dumps({"summary": summary, "msg_count": msg_count}, ensure_ascii=False)
        redis = await self._get_redis()
        if redis is not None:
            await redis.set(key, payload)
            await redis.expire(key, settings.redis.session_ttl)

    async def clear_summary(self, session_id: str) -> None:
        """清除摘要缓存"""
        key = f"{_SUMMARY_KEY_PREFIX}{session_id}"
        redis = await self._get_redis()
        if redis is not None:
            await redis.delete(key)


conversation_history = ConversationHistory()
