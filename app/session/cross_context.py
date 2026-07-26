"""跨设备上下文管理器

当 1 楼数字人回答"胃镜在7楼"时，自动生成跨设备上下文写入 Redis。
7 楼数字人在处理新对话时，检查 Redis 中是否有来自其他设备的上下文，注入 LLM prompt。

设计决策：
- 设备级上下文（不识别个人）：cross_ctx:{scene_id}:{target_device_id}
- TTL 5 分钟自动过期
- 消费后删除（一次性注入）
- 从回复中提取楼层关键词，推断目标设备 ID
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from app.infrastructure.redis import RedisPool

logger = logging.getLogger(__name__)

# Key 前缀
_KEY_PREFIX = "cross_ctx:"
# 默认 TTL（秒）
_DEFAULT_TTL = 300  # 5 分钟

# 楼层关键词提取正则
_FLOOR_PATTERN = re.compile(r"(\d+)\s*[楼层层]")


class CrossContextManager:
    """跨设备上下文管理器"""

    async def generate(
        self,
        scene_id: str,
        device_id: str,
        device_location: str,
        user_question: str,
        reply: str,
    ) -> None:
        """生成跨设备上下文

        从回复中提取楼层关键词，推断目标设备 ID，写入 Redis。
        如果回复中未提及其他楼层/区域，则不生成上下文。

        Args:
            scene_id: 场景 ID
            device_id: 当前设备 ID
            device_location: 当前设备位置描述
            user_question: 用户提问
            reply: 数字人回复
        """
        if not scene_id or not reply:
            return

        # 从回复中提取目标楼层
        target_floors = _FLOOR_PATTERN.findall(reply)
        if not target_floors:
            return

        # 排除当前设备所在楼层
        current_floor_match = _FLOOR_PATTERN.search(device_location or "")
        current_floor = current_floor_match.group(1) if current_floor_match else None

        for floor in target_floors:
            if floor == current_floor:
                continue

            # 推断目标设备 ID：scene_id + "_" + floor + "f"
            # 例如：hospital_7f, museum_2f
            target_device_id = f"{scene_id}_{floor}f"

            context_data = {
                "from_device_id": device_id,
                "from_device_location": device_location,
                "user_question": user_question,
                "from_reply": reply,
                "target_floor": floor,
                "target_device_id": target_device_id,
                "created_at": datetime.now().isoformat(),
            }

            # 写入 Redis：key = cross_ctx:{scene_id}:{target_device_id}
            key = f"{_KEY_PREFIX}{scene_id}:{target_device_id}"
            try:
                redis = await RedisPool.get()
                await redis.set(key, json.dumps(context_data, ensure_ascii=False), ex=_DEFAULT_TTL)
                logger.info(
                    "跨设备上下文已生成: %s → %s, question='%s'",
                    device_id, target_device_id, user_question[:30],
                )
            except Exception as e:
                logger.error("跨设备上下文写入失败: %s", e)

    async def get(self, scene_id: str, device_id: str) -> dict | None:
        """读取并消费跨设备上下文

        读取后立即删除（一次性注入），避免重复注入。

        Args:
            scene_id: 场景 ID
            device_id: 当前设备 ID

        Returns:
            上下文数据 dict，或 None
        """
        key = f"{_KEY_PREFIX}{scene_id}:{device_id}"
        try:
            redis = await RedisPool.get()
            data = await redis.get(key)
            if data:
                # 消费后删除
                await redis.delete(key)
                context = json.loads(data)
                logger.info(
                    "跨设备上下文已消费: device=%s, from=%s, question='%s'",
                    device_id, context.get("from_device_id"), context.get("user_question", "")[:30],
                )
                return context
        except Exception as e:
            logger.error("跨设备上下文读取失败: %s", e)
        return None

    def format_prompt(self, context: dict) -> str:
        """格式化跨设备上下文为 LLM prompt 文本

        Args:
            context: 跨设备上下文数据

        Returns:
            注入 system prompt 的文本
        """
        from_location = context.get("from_device_location", "其他区域")
        question = context.get("user_question", "")
        reply = context.get("from_reply", "")

        return (
            f"\n\n[跨设备上下文]\n"
            f"以下用户刚从{from_location}来到本设备位置，"
            f"之前在{from_location}问了「{question}」，"
            f"那边回复了「{reply}」。\n"
            f"请基于此上下文，为本用户提供更精准的指引。"
        )


# 单例
cross_context = CrossContextManager()
