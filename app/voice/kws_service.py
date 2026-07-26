"""sherpa-onnx 关键词检测（KWS）服务 — 替代 Whisper tiny + pypinyin 的唤醒词检测。

原理：专用 KWS 模型持续监听音频流，命中关键词时直接返回，无需通用 ASR 转写再字符串匹配。
优势：
  - 延迟 <30ms（vs Whisper tiny 631ms~6.5s）
  - 无幻觉/乱码（KWS 只检测预设关键词，不做通用转写）
  - 模型仅 ~5MB（int8），CPU 单线程即可实时
  - Apache 2.0 许可，100 台设备零授权费

用法：
  from app.voice.kws_service import kws_service
  matched, keyword, ms = await kws_service.detect(pcm_bytes)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from app.config.settings import settings

logger = logging.getLogger(__name__)

# 模型目录（相对于 python-ai-engine/）
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "kws" / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"


class KwsService:
    """唤醒词关键词检测服务（sherpa-onnx KeywordSpotter）"""

    def __init__(self) -> None:
        self._kws = None
        self._loaded = False

    def _load(self) -> None:
        """延迟加载 KWS 模型（首次调用时加载，约 5MB int8）"""
        if self._loaded:
            return

        import sherpa_onnx

        encoder = _MODEL_DIR / "encoder-epoch-99-avg-1-chunk-16-left-64.onnx"
        decoder = _MODEL_DIR / "decoder-epoch-99-avg-1-chunk-16-left-64.onnx"
        joiner = _MODEL_DIR / "joiner-epoch-99-avg-1-chunk-16-left-64.onnx"
        tokens = _MODEL_DIR / "tokens.txt"
        keywords = _MODEL_DIR / "keywords_custom.txt"

        # 校验文件存在
        for f in (encoder, decoder, joiner, tokens, keywords):
            if not f.exists():
                raise FileNotFoundError(f"KWS 模型文件缺失: {f}")

        logger.info("正在加载 KWS 唤醒模型: %s", _MODEL_DIR.name)
        t0 = time.monotonic()

        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            keywords_file=str(keywords),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            keywords_score=3.0,
            keywords_threshold=0.01,
            provider="cpu",
        )

        load_ms = int((time.monotonic() - t0) * 1000)
        self._loaded = True
        logger.info("KWS 唤醒模型加载完成: %dms", load_ms)

    async def detect(self, pcm_data: bytes, sample_rate: int = 16000) -> tuple[bool, str, int]:
        """检测 PCM 音频段中是否包含唤醒关键词。

        Args:
            pcm_data: 16-bit 单声道 PCM 原始字节
            sample_rate: 采样率（默认 16000）

        Returns:
            (matched, keyword, latency_ms)
            - matched: 是否命中关键词
            - keyword: 命中的关键词文本（未命中为空串）
            - latency_ms: 检测耗时（毫秒）
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load)

        t0 = time.monotonic()
        matched, keyword = await loop.run_in_executor(
            None, self._detect_sync, pcm_data, sample_rate
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        return matched, keyword, latency_ms

    def _detect_sync(self, pcm_data: bytes, sample_rate: int) -> tuple[bool, str]:
        """同步检测（在线程池中执行）"""
        # PCM 16-bit → float32 归一化
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # ── 重采样：浏览器 AudioContext 实际采样率常为 44100/48000，KWS 模型固定 16kHz ──
        target_rate = 16000
        if sample_rate != target_rate:
            duration = len(samples) / sample_rate
            target_len = int(duration * target_rate)
            if target_len > 0:
                # 线性插值重采样（轻量，无需 scipy）
                x_old = np.linspace(0, duration, len(samples), endpoint=False)
                x_new = np.linspace(0, duration, target_len, endpoint=False)
                samples = np.interp(x_new, x_old, samples).astype(np.float32)
                logger.debug("KWS 重采样: %dHz→%dHz, %d→%d samples", sample_rate, target_rate, len(x_old), target_len)

        # 尾部补零：让模型有足够上下文完成最后一个 token 的判定
        tail_paddings = np.zeros(int(0.66 * target_rate), dtype=np.float32)

        stream = self._kws.create_stream()
        stream.accept_waveform(target_rate, samples)
        stream.accept_waveform(target_rate, tail_paddings)
        stream.input_finished()

        detected_keyword = ""
        while self._kws.is_ready(stream):
            self._kws.decode_stream(stream)
            r = self._kws.get_result(stream)
            if r:
                detected_keyword = r
                break  # 命中即停，无需继续解码

        # 调试：未命中时保存音频供离线分析（使用系统临时目录，兼容 Windows）
        if not detected_keyword and len(samples) > 0:
            import wave as _wave
            import tempfile
            _dbg_dir = Path(tempfile.gettempdir()) / "dh_kws_dbg"
            _dbg_dir.mkdir(exist_ok=True)
            _dbg_files = sorted(_dbg_dir.glob("*.wav"))
            _idx = len(_dbg_files)
            _dbg_path = _dbg_dir / f"wake_{_idx:03d}.wav"
            # 保存为重采样后的 16kHz WAV（与模型输入一致，便于离线复现）
            pcm16 = (samples * 32767).astype(np.int16).tobytes()
            with _wave.open(str(_dbg_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_rate)
                wf.writeframes(pcm16)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            logger.info("KWS 未命中，已保存调试音频: %s (%.2fs, rms=%.4f, 原始采样率=%d)",
                        _dbg_path.name, len(samples) / target_rate, rms, sample_rate)

        return bool(detected_keyword), detected_keyword


# 全局单例
kws_service = KwsService()
