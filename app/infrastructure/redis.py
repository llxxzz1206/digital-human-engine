from __future__ import annotations

import logging

import redis.asyncio as aioredis

from app.config.settings import settings

logger = logging.getLogger(__name__)


class RedisPool:
    """Redis 异步连接池单例"""

    _instance: aioredis.Redis | None = None

    @classmethod
    async def get(cls) -> aioredis.Redis:
        """获取 Redis 连接实例，首次调用时初始化连接池"""
        if cls._instance is None:
            cls._instance = aioredis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password or None,
                max_connections=settings.redis.pool_size,
                decode_responses=True,
                protocol=2,  # Redis 5.0.x 兼容，不支持 RESP3 的 HELLO 命令
            )
            logger.info(
                "Redis 连接池已初始化: %s:%d/%d, pool_size=%d",
                settings.redis.host,
                settings.redis.port,
                settings.redis.db,
                settings.redis.pool_size,
            )
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        """关闭 Redis 连接池"""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None
            logger.info("Redis 连接池已关闭")

    @classmethod
    async def ping(cls) -> bool:
        """检查 Redis 连通性"""
        try:
            client = await cls.get()
            return await client.ping()
        except Exception as e:
            logger.error("Redis ping 失败: %s", e)
            return False
