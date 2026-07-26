from __future__ import annotations

import asyncio
import logging
import struct
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from app.config.settings import settings

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(self) -> None:
        self._model = None
        self._wake_model = None
        self._provider = settings.asr.provider
        self._model_name = settings.asr.model
        self._wake_model_name = settings.asr.wake_model
        self._device = settings.asr.device
        self._loaded = False
        self._wake_loaded = False

    def _load_model(self) -> None:
        """延迟加载 Whisper 模型（首次调用时加载，仅 provider=whisper 时）"""
        if self._loaded or self._provider != "whisper":
            return
        import whisper
        logger.info("正在加载 Whisper 模型: %s, device=%s ...", self._model_name, self._device)
        self._model = whisper.load_model(self._model_name, device=self._device)
        self._loaded = True
        logger.info("Whisper 模型加载完成: %s", self._model_name)

    def _load_wake_model(self) -> None:
        """延迟加载唤醒专用 Whisper 模型（更小、更快）。

        唤醒词检测永远走本地 Whisper tiny，不受 ASR provider 影响：
        对话识别可上云（xunfei），但唤醒是常驻监听，本地小模型免费且低延迟。
        """
        if self._wake_loaded:
            return
        import whisper
        logger.info("正在加载 Whisper 唤醒模型: %s, device=%s ...", self._wake_model_name, self._device)
        self._wake_model = whisper.load_model(self._wake_model_name, device=self._device)
        self._wake_loaded = True
        logger.info("Whisper 唤醒模型加载完成: %s", self._wake_model_name)

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, bits: int = 16, channels: int = 1) -> bytes:
        """将原始 PCM 数据添加 WAV 头"""
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        data_size = len(pcm_data)
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # chunk size
            1,   # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
            b'data',
            data_size,
        )
        return header + pcm_data

    async def transcribe(
        self,
        audio: bytes,
        format: str = "wav",
        wake: bool = False,
        initial_prompt: str | None = None,
    ) -> str:
        """语音转文字

        Args:
            audio: 音频二进制数据
            format: 音频格式 (wav/mp3/flac 等)
            wake: True 时使用唤醒专用小模型（更快），否则用对话模型（更准）
            initial_prompt: 提示词（如唤醒暗号），喂给 Whisper 偏向识别这些字词，
                对短音频/专有词识别率提升明显

        Returns:
            识别出的文本
        """
        loop = asyncio.get_event_loop()
        if wake:
            await loop.run_in_executor(None, self._load_wake_model)
            model = self._wake_model
        else:
            await loop.run_in_executor(None, self._load_model)
            model = self._model

        # 如果是原始 PCM，添加 WAV 头
        if format == "pcm":
            audio = self._pcm_to_wav(audio, sample_rate=16000, bits=16, channels=1)
            format = "wav"

        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio)
            tmp_path = f.name

        try:
            transcribe_kwargs = {"language": "zh"}
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(tmp_path, **transcribe_kwargs),
            )
            text = result.get("text", "").strip()
            logger.info("ASR 识别结果%s: text=%s (len=%d)", "（唤醒）" if wake else "", text[:50], len(text))
            return text
        except Exception as e:
            logger.error("ASR 识别失败: %s", e)
            return ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        session_id: str = "",
        send_func=None,
    ) -> str:
        """流式语音识别：接收音频分片，发送中间结果，返回最终文本

        Args:
            audio_chunks: 音频分片异步生成器
            session_id: 会话 ID
            send_func: 中间结果发送函数 async func(msg: dict) -> None

        Returns:
            最终识别文本
        """
        # 收集所有音频分片
        chunks = []
        chunk_count = 0
        async for chunk in audio_chunks:
            chunks.append(chunk)
            chunk_count += 1

            # 每收到 5 个分片，发送一次中间结果
            if send_func and chunk_count % 5 == 0:
                await send_func({
                    "type": "asr.partial",
                    "payload": {
                        "sessionId": session_id,
                        "text": f"[接收中...已收到{chunk_count}个音频分片]",
                        "chunkCount": chunk_count,
                        "done": False,
                    },
                })

        if not chunks:
            return ""

        # 合并所有分片
        full_audio = b"".join(chunks)
        logger.info("ASR 流式: session_id=%s, chunks=%d, total_bytes=%d", session_id, chunk_count, len(full_audio))

        # 发送"正在识别"中间结果
        if send_func:
            await send_func({
                "type": "asr.partial",
                "payload": {
                    "sessionId": session_id,
                    "text": "[正在识别...]",
                    "chunkCount": chunk_count,
                    "done": False,
                },
            })

        # 执行识别（full_audio 是原始 PCM，transcribe 内部会自动加 WAV 头）
        result = await self.transcribe(full_audio, "pcm")

        # 发送最终结果
        if send_func:
            await send_func({
                "type": "asr.result",
                "payload": {
                    "sessionId": session_id,
                    "text": result,
                    "isFinal": True,
                },
            })

        return result


asr_service = ASRService()
