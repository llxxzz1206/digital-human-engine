from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 默认动作集（Redis 未配置时的兜底）
DEFAULT_GESTURES = {"idle", "talking", "greeting", "point_left", "point_right", "bow"}
DEFAULT_LOOP_STATES = {"idle", "talking"}


class AvatarDriver:
    """Avatar 驱动器：纯状态驱动，告诉前端切换播放哪个视频

    前端收到 state 后切换对应视频文件，无需额外时间轴或口型数据。
    视频本身已包含完整的动作和口型信息。

    动作列表从形象配置（Redis）读取，支持动态扩展。
    """

    def __init__(self, gestures: set[str] | None = None, loop_states: set[str] | None = None):
        self._gestures = gestures or DEFAULT_GESTURES
        self._loop_states = loop_states or DEFAULT_LOOP_STATES

    @classmethod
    def from_config(cls, avatar_config: dict) -> "AvatarDriver":
        """从 Redis 形象配置构造 driver

        avatar_config 中读取:
          gestures: "idle,talking,greeting,point_left,point_right,bow,wave" (逗号分隔)
          loopStates: "idle,talking" (逗号分隔)
        缺失字段用默认值。
        """
        gestures = DEFAULT_GESTURES
        loop_states = DEFAULT_LOOP_STATES

        raw_gestures = avatar_config.get("gestures", "")
        if raw_gestures:
            gestures = {g.strip() for g in raw_gestures.split(",") if g.strip()}

        raw_loop = avatar_config.get("loopStates", "")
        if raw_loop:
            loop_states = {s.strip() for s in raw_loop.split(",") if s.strip()}

        return cls(gestures=gestures, loop_states=loop_states)

    async def generate_drive(self, state: str, text: str = "") -> dict[str, Any]:
        """生成 Avatar 驱动数据

        Args:
            state: 视频状态 (idle/talking/greeting/point_left/point_right/bow/自定义)
            text:  回复文本。当前仅做状态切换、未消费此参数；
                   保留为后续口型同步/视频驱动预留，勿删（调用方均按位置传入）。

        Returns:
            {"state": "talking", "loop": true}
        """
        if state not in self._gestures:
            logger.warning("未知 avatar 状态: %s, 降级到 talking", state)
            state = "talking"

        return {
            "state": state,
            "loop": state in self._loop_states,
        }


# 默认实例（向后兼容，未传形象配置时使用）
avatar_driver = AvatarDriver()
