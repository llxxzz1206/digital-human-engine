"""会话级检索缓存 - 针对追问场景优化"""
from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


class SessionSearchCache:
    """会话级检索缓存（只缓存最近一次检索结果）
    
    设计目标：
    1. 针对追问场景：用户连续提问时复用检索结果
    2. 极低延迟：纯内存 + 规则判断，无 ML 调用
    3. 简单有效：时间窗口 + 追问关键词检测
    
    使用场景：
    - 用户问："二楼有什么展品" → RAG 检索
    - 用户追问："三楼呢" → 复用检索结果（追问意图检测）
    - 用户问："今天开门吗" → 新检索（时间超限或问题不相关）
    """

    # 追问关键词（微秒级匹配）
    FOLLOWUP_PATTERNS = ["呢", "那", "怎么样", "还有呢", "然后呢", "那呢"]
    
    # 追问问题长度阈值（短问题更可能是追问）
    FOLLOWUP_MAX_LENGTH = 15
    
    # 时间窗口（秒）
    TIME_WINDOW = 30

    def __init__(self) -> None:
        # session_id -> (query, (results, source_type), timestamp)
        self._last_search: dict[str, tuple[str, tuple[list[dict], str], float]] = {}

    def is_followup_question(self, new_query: str, last_query: str | None) -> bool:
        """检测追问意图（纯规则，无 ML 调用）
        
        规则：
        1. 新问题包含追问词："呢"、"那"、"怎么样"
        2. 新问题较短（<15字）且包含追问特征
        3. 上一个问题不为空
        
        性能：微秒级，无网络/计算开销
        """
        if not last_query:
            return False
        
        # 规则 1：追问关键词
        for pattern in self.FOLLOWUP_PATTERNS:
            if pattern in new_query:
                return True
        
        # 规则 2：短问题 + 代词
        if len(new_query) < self.FOLLOWUP_MAX_LENGTH:
            pronouns = ["它", "这", "那", "这个", "那个"]
            if any(p in new_query for p in pronouns):
                return True
        
        return False

    async def get_or_search(
        self,
        session_id: str,
        query: str,
        search_fn: Callable[[], tuple[list[dict], str]],
    ) -> tuple[list[dict], str]:
        """获取缓存或执行检索
        
        Args:
            session_id: 会话 ID
            query: 用户问题
            search_fn: 检索函数（异步），返回 (results, source_type)
            
        Returns:
            (检索结果列表, 来源类型)
        """
        now = time.time()
        
        # 1. 检查是否有缓存
        if session_id in self._last_search:
            last_query, (results, source_type), timestamp = self._last_search[session_id]
            
            # 2. 时间窗口：30秒内
            if now - timestamp < self.TIME_WINDOW:
                # 3. 追问检测
                if self.is_followup_question(query, last_query):
                    logger.info(
                        "命中追问缓存: session=%s, last=%s, new=%s",
                        session_id, last_query[:20], query[:20]
                    )
                    return results, source_type
        
        # 4. 执行检索并缓存
        result = await search_fn()
        self._last_search[session_id] = (query, result, now)
        logger.debug("检索结果已缓存: session=%s, query=%s", session_id, query[:20])
        return result

    def clear_session(self, session_id: str) -> None:
        """清除会话缓存（会话结束时调用）"""
        if session_id in self._last_search:
            del self._last_search[session_id]

    def clear_expired(self, max_age: float = 300) -> int:
        """清理过期缓存（后台任务调用）
        
        Args:
            max_age: 最大缓存时间（秒），默认 5 分钟
            
        Returns:
            清理的记录数
        """
        now = time.time()
        expired = [
            sid for sid, (_, _, ts) in self._last_search.items()
            if now - ts > max_age
        ]
        for sid in expired:
            del self._last_search[sid]
        
        if expired:
            logger.info("清理过期检索缓存: %d 条", len(expired))
        
        return len(expired)


# 全局实例
search_cache = SessionSearchCache()