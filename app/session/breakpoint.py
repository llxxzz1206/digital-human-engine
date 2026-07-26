"""打断续接管理器：保存讲解断点，支持用户插话后从断点续讲

展馆场景需求：
  - 数字人正在讲解展品："这件青铜器的纹饰是饕餮纹，象征着威严与神秘，常用于祭祀场合..."
  - 用户插话："等等，饕餮纹是什么"
  - 系统回答追问，然后从断点续讲："...饕餮纹是商周时期常见的纹饰，多见于青铜礼器。回到刚才的讲解，这件青铜器..."

实现方式：
  1. interrupt 时保存当前回复的断点（session_id -> {text, position, context}）
  2. 回复生成时检查是否有断点，有则续讲，无则正常生成
  3. 断点 TTL 5分钟（避免跨会话污染）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 断点 TTL（秒）
BREAKPOINT_TTL = 300


@dataclass
class Breakpoint:
    """讲解断点"""
    text: str                  # 完整讲解文本
    position: int              # 断点字符位置
    context: dict[str, Any]    # 上下文（展品ID、话题等）
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查断点是否过期"""
        return time.time() - self.created_at > BREAKPOINT_TTL
    
    def get_remaining_text(self) -> str:
        """获取断点后的剩余文本"""
        if self.position >= len(self.text):
            return ""
        return self.text[self.position:]
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "position": self.position,
            "context": self.context,
            "created_at": self.created_at,
        }


class BreakpointManager:
    """断点管理器：管理所有会话的讲解断点"""
    
    def __init__(self) -> None:
        self._breakpoints: dict[str, Breakpoint] = {}  # session_id -> Breakpoint
    
    def save(self, session_id: str, text: str, position: int, context: dict | None = None) -> None:
        """保存讲解断点
        
        Args:
            session_id: 会话ID
            text: 完整讲解文本
            position: 断点字符位置（用户打断时的位置）
            context: 上下文信息（如展品ID、话题等）
        """
        if not text or position < 0:
            return
        
        # 确保position不超过文本长度
        position = min(position, len(text))
        
        self._breakpoints[session_id] = Breakpoint(
            text=text,
            position=position,
            context=context or {},
        )
        logger.info("保存讲解断点: session_id=%s, position=%d/%d, remaining=%d字符",
                    session_id, position, len(text), len(text) - position)
    
    def get(self, session_id: str) -> Breakpoint | None:
        """获取并检查断点（已过期则删除）"""
        bp = self._breakpoints.get(session_id)
        if bp is None:
            return None
        
        if bp.is_expired():
            self.remove(session_id)
            logger.info("讲解断点已过期: session_id=%s", session_id)
            return None
        
        return bp
    
    def remove(self, session_id: str) -> None:
        """删除断点"""
        if session_id in self._breakpoints:
            del self._breakpoints[session_id]
    
    def consume(self, session_id: str) -> Breakpoint | None:
        """消费断点（获取后立即删除）"""
        bp = self.get(session_id)
        if bp:
            self.remove(session_id)
            logger.info("消费讲解断点: session_id=%s", session_id)
        return bp
    
    def clear_expired(self) -> int:
        """清理所有过期断点"""
        expired = [sid for sid, bp in self._breakpoints.items() if bp.is_expired()]
        for sid in expired:
            del self._breakpoints[sid]
        if expired:
            logger.info("清理过期断点: %d个", len(expired))
        return len(expired)
    
    def stats(self) -> dict:
        """统计信息"""
        return {
            "total_breakpoints": len(self._breakpoints),
            "sessions": list(self._breakpoints.keys()),
        }


# 全局实例
breakpoint_manager = BreakpointManager()