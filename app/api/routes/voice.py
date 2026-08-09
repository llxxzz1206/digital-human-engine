"""固定话术音频接口

数字人语音交互的固定话术（收尾语），预合成缓存，
运行时直接取缓存 PCM，绝不实时 TTS（省延迟、省配额、音色一致）。

- 收尾语 closer："您还有什么问题吗"，免唤醒追问窗口超时后播放

（填充语 filler / 衔接语 connector 机制已下线：实际体验中"稍等几秒"+"好啦"
打断感明显，已整体移除，回答直接播出。）

前端（交互控制器）启动时拉取一次并内存缓存，播放时直接喂 AudioPlayer。
"""
from __future__ import annotations

import base64
import logging

from fastapi import APIRouter

from app.rag.tts_cache import tts_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["语音话术"])

# 固定话术（文本即缓存键，勿随意改动，改了会触发重新合成）
FIXED_PHRASES = {
    "closer": "您还有什么问题吗",
}


async def warm_fixed_phrases() -> None:
    """预热三句固定话术到 TTS 缓存（启动时后台调用，首次合成、之后命中）"""
    for key, text in FIXED_PHRASES.items():
        try:
            audio = await tts_cache.get_or_synthesize(text)
            logger.info("固定话术预热: %s='%s' → %d bytes", key, text, len(audio))
        except Exception as e:
            logger.warning("固定话术预热失败（运行时将惰性合成）: %s, error=%s", key, e)


@router.get("/fixed-phrases")
async def get_fixed_phrases():
    """返回固定话术的 base64 PCM 音频（缓存未命中则现场合成并缓存）"""
    data = {}
    for key, text in FIXED_PHRASES.items():
        try:
            audio = await tts_cache.get_or_synthesize(text)
            # 16kHz 16bit 单声道 → 每秒 32000 字节
            duration_ms = int(len(audio) / 32000 * 1000) if audio else 0
            data[key] = {
                "text": text,
                "audio": base64.b64encode(audio).decode("ascii") if audio else "",
                "durationMs": duration_ms,
            }
        except Exception as e:
            logger.error("固定话术获取失败: %s, error=%s", key, e)
            data[key] = {"text": text, "audio": "", "durationMs": 0}
    return {"code": 200, "data": data}
