"""TTS 预缓存服务：常用语预合成，降低首字延迟

展馆场景常用语：
  - 问候语："您好，有什么可以帮您？"
  - 过渡语："请看这边"、"这是..."
  - 感谢语："不客气"、"很高兴为您服务"
  
原理：
  - 服务启动时预合成这些句子，存入 TTS 缓存表
  - 用户触发时直接从缓存读取，延迟从 200-400ms 降至 <10ms
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config.settings import settings
from app.rag.tts_cache import tts_cache

logger = logging.getLogger(__name__)

# 展馆场景常用语（按场景分类）
COMMON_PHRASES: dict[str, list[str]] = {
    "greeting": [
        "您好，有什么可以帮您？",
        "您好，我是数字人助手小诺，请问有什么可以帮您？",
        "您好，欢迎来到展馆。",
    ],
    "transition": [
        "请看这边。",
        "这边请。",
        "请跟我来。",
        "这件展品是",
        "这是",
    ],
    "thanks": [
        "不客气。",
        "很高兴为您服务。",
        "希望能帮到您。",
    ],
    "unknown": [
        "不好意思没听清，请再说一遍。",
        "抱歉，我没太听懂，能换个说法吗？",
        "这个问题我暂时回答不了，稍后会有工作人员为您解答。",
    ],
    "goodbye": [
        "再见，祝您参观愉快。",
        "感谢您的参观，再见。",
    ],
}


async def pre_cache_common_phrases() -> dict[str, Any]:
    """预缓存常用语
    
    Returns:
        统计信息 {"total": 10, "cached": 8, "failed": 2}
    """
    if not settings.tts.app_id:
        logger.warning("TTS 未配置，跳过预缓存")
        return {"total": 0, "cached": 0, "failed": 0, "reason": "TTS not configured"}
    
    total = 0
    cached = 0
    failed = 0
    
    # 展开所有短语
    all_phrases: list[str] = []
    for category, phrases in COMMON_PHRASES.items():
        all_phrases.extend(phrases)
    
    total = len(all_phrases)
    logger.info("开始预缓存常用语: %d条", total)
    
    for phrase in all_phrases:
        try:
            # 调用 tts_cache 的 get_or_synthesize（会检查缓存）
            pcm = await tts_cache.get_or_synthesize(phrase)
            if pcm:
                cached += 1
                logger.debug("预缓存成功: '%s'", phrase[:20])
            else:
                failed += 1
                logger.warning("预缓存失败(空结果): '%s'", phrase[:20])
        except Exception as e:
            failed += 1
            logger.error("预缓存异常: '%s', error=%s", phrase[:20], e)
    
    logger.info("预缓存完成: total=%d, cached=%d, failed=%d", total, cached, failed)
    return {"total": total, "cached": cached, "failed": failed}


async def pre_cache_for_scene(scene_id: str) -> dict[str, Any]:
    """为特定场景预缓存（如博物馆、医院）
    
    Args:
        scene_id: 场景ID (hospital/museum/...)
    
    Returns:
        统计信息
    """
    # 场景特定短语（可从数据库加载）
    scene_phrases: dict[str, list[str]] = {
        "hospital": [
            "挂号处在一楼大厅。",
            "急诊科在一楼东侧。",
            "请到二楼内科门诊。",
        ],
        "museum": [
            "这件青铜器来自商代。",
            "这幅画创作于明朝。",
            "请到三楼书画展厅。",
        ],
    }
    
    phrases = scene_phrases.get(scene_id, [])
    if not phrases:
        return {"total": 0, "cached": 0, "failed": 0, "reason": f"no phrases for scene {scene_id}"}
    
    cached = 0
    failed = 0
    
    for phrase in phrases:
        try:
            pcm = await tts_cache.get_or_synthesize(phrase)
            if pcm:
                cached += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    
    return {"total": len(phrases), "cached": cached, "failed": failed}


# ── 预缓存任务（服务启动时调用） ──

_pre_cache_task: asyncio.Task | None = None


def start_pre_cache_task() -> None:
    """启动预缓存任务（服务启动时调用）"""
    global _pre_cache_task
    
    async def _run():
        # 延迟 5 秒启动（等 TTS 服务初始化完成）
        await asyncio.sleep(5)
        await pre_cache_common_phrases()
    
    if _pre_cache_task is None or _pre_cache_task.done():
        _pre_cache_task = asyncio.create_task(_run())
        logger.info("TTS 预缓存任务已启动")


def stop_pre_cache_task() -> None:
    """停止预缓存任务"""
    global _pre_cache_task
    if _pre_cache_task and not _pre_cache_task.done():
        _pre_cache_task.cancel()
        logger.info("TTS 预缓存任务已停止")