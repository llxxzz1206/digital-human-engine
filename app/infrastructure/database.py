from __future__ import annotations

import logging

import asyncpg

from app.config.settings import settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """PostgreSQL 异步连接池单例（类似 RedisPool 模式）"""

    _pool: asyncpg.Pool | None = None

    @classmethod
    async def get(cls) -> asyncpg.Pool:
        """获取连接池，首次调用时初始化"""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                settings.database.url,
                min_size=2,
                max_size=10,
            )
            logger.info("PostgreSQL 连接池已初始化: %s", settings.database.url.split("@")[-1])
        return cls._pool

    @classmethod
    async def close(cls) -> None:
        """关闭连接池"""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("PostgreSQL 连接池已关闭")

    @classmethod
    async def ping(cls) -> bool:
        """检查数据库连通性"""
        try:
            pool = await cls.get()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error("PostgreSQL ping 失败: %s", e)
            return False

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        """快捷执行 SQL"""
        pool = await cls.get()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args) -> list[asyncpg.Record]:
        """快捷查询多行"""
        pool = await cls.get()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args) -> asyncpg.Record | None:
        """快捷查询单行"""
        pool = await cls.get()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query: str, *args) -> any:
        """快捷查询单值"""
        pool = await cls.get()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)
