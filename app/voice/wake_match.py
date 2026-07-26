"""唤醒词拼音匹配

ASR 对唤醒词的识别常有同音字误差（如"小医"→"小姨"、"小一"），
按汉字精确包含匹配会漏触发。这里把识别文本和唤醒词都转成**无调拼音**
再做包含匹配：小医 / 小姨 / 小一 / 小依 的拼音都是 "xiao yi"，统一命中。

增强容错（2026-07）：
 - 去连续重复音节：口吃/环境噪声导致"小小牛牛"→"小牛"
 - 子序列匹配：允许关键词音节之间插入多余音节（如"你好小小牛"中多了一个 xiao）
 - 部分匹配：关键词 ≥75% 音节按序出现即命中（容忍丢字，如"好，小牛"丢了"你"）

为什么用拼音而非编辑距离：
 - 中文唤醒误识几乎全是同音替换，拼音天然对齐这类误差，规则简单可解释；
 - pypinyin 纯 Python、无模型、毫秒级，100 台大屏部署无额外负担。

音节之间用空格分隔，避免跨音节误配（如 "xi"+"an" 误拼成 "xian"）。
"""
from __future__ import annotations

import logging

from pypinyin import lazy_pinyin

logger = logging.getLogger(__name__)


def _pinyin_key(text: str) -> str:
    """归一化成拼音键：中文转无调拼音，保留英文/数字，丢弃标点空白，音节空格分隔"""
    parts = lazy_pinyin(text.lower())
    # lazy_pinyin 对标点原样返回（如 "，"），用 isalnum 过滤掉
    return " ".join(p for p in parts if p.isalnum())


def _dedup_syllables(syllables: list[str]) -> list[str]:
    """去除连续重复音节（处理口吃/结巴：小小牛牛 → 小牛）"""
    if not syllables:
        return []
    result = [syllables[0]]
    for s in syllables[1:]:
        if s != result[-1]:
            result.append(s)
    return result


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """判断 needle 是否为 haystack 的子序列（按序出现，允许中间插入其他音节）"""
    it = iter(haystack)
    return all(s in it for s in needle)


def _partial_match(needle: list[str], haystack: list[str], threshold: float = 0.75) -> bool:
    """部分匹配：needle 中 ≥threshold 比例的音节按序出现在 haystack 中。
    未命中的音节跳过（不推进搜索位置），允许丢字。"""
    if not needle:
        return False
    matched = 0
    hi = 0
    for n in needle:
        for j in range(hi, len(haystack)):
            if haystack[j] == n:
                matched += 1
                hi = j + 1
                break
        # 未找到则跳过该音节，hi 不动，后续音节仍可从当前位置匹配
    return matched / len(needle) >= threshold


def match_wake(text: str, phrases: list[str]) -> bool:
    """判断 ASR 文本是否命中任一唤醒词（多层容错匹配）

    匹配策略（从严到宽）：
    1. 精确拼音子串匹配（原始逻辑）
    2. 去重复音节后子串匹配（处理口吃："你好小小牛牛" → "你好小牛"）
    3. 子序列匹配（允许插入多余音节）
    4. 部分匹配（≥75% 音节按序出现，容忍丢字："好，小牛" 丢了"你"）

    Args:
        text: ASR 识别文本
        phrases: 唤醒词列表（如 ["你好小牛", "你好小小牛"]）

    Returns:
        任一唤醒词命中即返回 True
    """
    if not text or not phrases:
        return False
    text_key = _pinyin_key(text)
    if not text_key:
        return False

    text_syllables = text_key.split()
    text_deduped = _dedup_syllables(text_syllables)

    for phrase in phrases:
        key = _pinyin_key(phrase)
        if not key:
            continue
        phrase_syllables = key.split()

        # 策略 1：精确子串
        if key in text_key:
            return True

        # 策略 2：去重复后子串
        deduped_key = " ".join(_dedup_syllables(phrase_syllables))
        deduped_text = " ".join(text_deduped)
        if deduped_key in deduped_text:
            return True

        # 策略 3：子序列匹配（关键词音节按序出现在文本中）
        if _is_subsequence(phrase_syllables, text_deduped):
            return True

        # 策略 4：部分匹配（≥75% 音节按序出现）
        if len(phrase_syllables) >= 3 and _partial_match(phrase_syllables, text_deduped, 0.75):
            return True

    return False
