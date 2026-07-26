"""噪音过滤器（Stage 4b）：多层过滤 Whisper 幻觉 / 底噪 / 无关输入

层级（由前到后，任一层命中即短路）：
  L1 VAD 门控  — 音频 RMS 能量过低 → 不送 ASR（在 _handle_audio_end 中调用）
  黑名单       — ASR 文本命中 Whisper 高频幻觉词 → 不送 LLM（在 _handle_audio_end 中调用）
  L2 Rerank   — 检索重排最高分低于阈值 → 不送 LLM（在 route_by_score 中调用）
  L3 LLM 兜底 — system prompt 指令：无意义输入回复"没听清"（在 hospital_skill 中配置）
"""

from __future__ import annotations

import logging
import math
import re
import struct

logger = logging.getLogger(__name__)

# ── Whisper 高频幻觉词（静音/底噪上常见） ──
# 来源：实测 + 社区报告。命中任一即判定为噪音，零成本。
_HALLUCINATION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"字幕",
        r"Zither",
        r"Harp",
        r"索兰娅",
        r"谢谢观看",
        r"感谢观看",
        r"字幕製作人",
        r"字幕制作",
        r"Transcrib",
        r"Subtitle",
        r"Amara\.org",
        r"by\s*索",
    ]
]

# 固定噪音回复文案（与前端 fixedPhrases 的 closer 同级，后续可预合成缓存）
NOISE_REPLY_TEXT = "不好意思没听清，请再说一遍"


def is_hallucination(text: str) -> bool:
    """检查 ASR 文本是否命中 Whisper 幻觉黑名单"""
    if not text or not text.strip():
        return True  # 空文本视为噪音
    for pat in _HALLUCINATION_PATTERNS:
        if pat.search(text):
            logger.info("噪音黑名单命中: text='%s', pattern='%s'", text[:40], pat.pattern)
            return True
    return False


def compute_rms(audio_bytes: bytes, sample_width: int = 2) -> float:
    """计算 PCM 音频的 RMS 能量（16-bit 小端，归一化到 0-1）

    Args:
        audio_bytes: 原始 PCM 字节
        sample_width: 采样位宽（字节），默认 2（16-bit）

    Returns:
        RMS 值（0.0 = 静音，1.0 = 满幅）
    """
    if not audio_bytes or len(audio_bytes) < sample_width:
        return 0.0

    n_samples = len(audio_bytes) // sample_width
    if n_samples == 0:
        return 0.0

    # 解包 16-bit 有符号整数
    fmt = f"<{n_samples}h"
    try:
        samples = struct.unpack(fmt, audio_bytes[: n_samples * sample_width])
    except struct.error:
        return 0.0

    sum_sq = sum(s * s for s in samples)
    rms = math.sqrt(sum_sq / n_samples) / 32768.0
    return rms
