from __future__ import annotations

import hashlib
import logging

from app.config.settings import settings
from app.infrastructure.database import DatabasePool
from app.voice.tts_service import tts_service

logger = logging.getLogger(__name__)

# 16kHz 16bit mono PCM 每秒字节数
PCM_BYTES_PER_SECOND = 16000 * 16 // 8 * 1


class TTSCacheService:
    """TTS 音频缓存服务

    将知识片段的 TTS 合成音频缓存到 PostgreSQL，
    命中时直接返回缓存数据，避免重复合成。
    """

    @staticmethod
    def _hash_text(text: str, speaker: str) -> str:
        """计算文本+发音人的 SHA256 哈希"""
        content = f"{speaker}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def ensure_table(self) -> None:
        """确保缓存表存在"""
        await DatabasePool.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_audio_cache (
                id SERIAL PRIMARY KEY,
                text_hash VARCHAR(64) NOT NULL,
                text_content TEXT NOT NULL,
                audio_data BYTEA NOT NULL,
                audio_format VARCHAR(10) DEFAULT 'pcm',
                speaker VARCHAR(50) DEFAULT 'x4_xiaoyan',
                duration_ms INTEGER,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(text_hash, speaker)
            )
        """)

    async def get(self, text: str, speaker: str | None = None) -> bytes | None:
        """查询缓存

        Args:
            text: 文本内容
            speaker: 发音人（默认使用配置中的发音人）

        Returns:
            缓存的 PCM 音频数据，未命中返回 None
        """
        speaker = speaker or settings.tts.voice
        text_hash = self._hash_text(text, speaker)

        row = await DatabasePool.fetchrow(
            "SELECT audio_data, duration_ms FROM knowledge_audio_cache "
            "WHERE text_hash = $1 AND speaker = $2",
            text_hash, speaker,
        )
        if row:
            logger.debug("TTS 缓存命中: text=%s..., duration=%dms", text[:20], row["duration_ms"] or 0)
            return row["audio_data"]
        return None

    async def put(
        self,
        text: str,
        audio_data: bytes,
        speaker: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """存入缓存

        Args:
            text: 文本内容
            audio_data: PCM 音频二进制数据
            speaker: 发音人
            duration_ms: 音频时长(ms)
        """
        speaker = speaker or settings.tts.voice
        text_hash = self._hash_text(text, speaker)

        await DatabasePool.execute(
            """
            INSERT INTO knowledge_audio_cache (text_hash, text_content, audio_data, audio_format, speaker, duration_ms)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (text_hash, speaker) DO UPDATE SET
                audio_data = EXCLUDED.audio_data,
                duration_ms = EXCLUDED.duration_ms,
                created_at = NOW()
            """,
            text_hash, text, audio_data, "pcm", speaker, duration_ms,
        )
        logger.info("TTS 缓存已存储: text=%s..., size=%d bytes", text[:20], len(audio_data))

    async def get_or_synthesize(self, text: str, speaker: str | None = None) -> bytes:
        """获取缓存或合成音频

        优先查缓存，未命中则调用 TTS 合成并缓存。

        Returns:
            PCM 音频二进制数据
        """
        # 1. 查缓存
        cached = await self.get(text, speaker)
        if cached is not None:
            return cached

        # 2. 调用 TTS 合成
        audio_chunks: list[bytes] = []
        async for chunk in tts_service.synthesize_stream(text):
            if chunk:
                audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        # 3. 存入缓存
        if audio_data:
            # 估算时长：16kHz 16bit mono → 每秒 32000 字节
            duration_ms = int(len(audio_data) / PCM_BYTES_PER_SECOND * 1000)
            await self.put(text, audio_data, speaker, duration_ms)

        return audio_data


tts_cache = TTSCacheService()
