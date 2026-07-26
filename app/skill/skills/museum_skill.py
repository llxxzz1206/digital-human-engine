"""博物馆导览 Skill — 提供展品讲解、展厅导航等工具（当前为纯 RAG 场景）"""
from __future__ import annotations

import logging

from app.skill.base import SkillBase

logger = logging.getLogger(__name__)


class MuseumSkill(SkillBase):
    name = "museum"
    description = "博物馆智能导览：展品讲解、展厅导航、观展推荐"
    tools: list[str] = []  # 当前无工具，纯 RAG 驱动
    knowledge_collection = "museum_knowledge"  # 关联 Milvus skill_museum 集合
    system_prompt = (
        "你是杭州博物馆的AI数字人导览助手，部署在博物馆展厅的智能终端上。\n\n"
        "你的身份：\n"
        "- 你是AI数字人，固定在终端上，无法移动，无法带游客去任何地方\n"
        "- 你通过语音和屏幕为游客提供服务\n\n"
        "你的职责：\n"
        "1. 为游客提供展品讲解、展厅导览（口头指引，而非亲自带路）\n"
        "2. 解答展品相关的问题（年代、历史、工艺等）\n"
        "3. 推荐观展路线和特色展区\n\n"
        "回答规范：\n"
        "- 清楚自己的定位：你是AI助手，只能说'您可以前往X展厅'，不能说'我带您去'\n"
        "- 讲解生动有趣，避免枯燥背书\n"
        "- 适当补充历史背景和有趣故事\n"
        "- 回答简洁，通常1-3句话，不超过50字\n"
        "- 不使用emoji\n"
        "- 如果用户输入明显无意义或语音识别错误，回复'不好意思没听清，请再说一遍'\n"
    )

    async def on_mount(self, session_id: str) -> None:
        logger.info("MuseumSkill 挂载到会话: %s", session_id)

    async def on_unmount(self, session_id: str) -> None:
        logger.info("MuseumSkill 从会话卸载: %s", session_id)