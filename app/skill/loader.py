from __future__ import annotations

import asyncio
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from app.skill.base import SkillBase

logger = logging.getLogger(__name__)


class RedisSkill(SkillBase):
    """从 Redis 配置加载的 Skill（动态配置）"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.name = config.get("id", "")
        self.description = config.get("description", "")
        self.tools = json.loads(config.get("tools", "[]")) if isinstance(config.get("tools"), str) else config.get("tools", [])
        self.knowledge_collection = config.get("knowledgeCollection", "")
        self.system_prompt = config.get("systemPrompt", "")
        self.scene = config.get("scene", "")
        self.status = config.get("status", "ENABLE")

    async def on_mount(self, session_id: str) -> None:
        logger.info("RedisSkill %s 挂载到会话: %s", self.name, session_id)

    async def on_unmount(self, session_id: str) -> None:
        logger.info("RedisSkill %s 从会话卸载: %s", self.name, session_id)


class SkillLoader:
    """Skill 加载器，负责动态加载和管理 Skill 插件"""

    def __init__(self) -> None:
        self._skills: dict[str, SkillBase] = {}
        self._tool_to_skill: dict[str, str] = {}  # tool_name -> skill_name 反向映射
        self._redis_loaded = False

    def load_skills(self, skills_dir: Path) -> None:
        """从指定目录加载所有 Skill（代码定义）"""
        if not skills_dir.exists():
            logger.warning("Skills 目录不存在: %s", skills_dir)
            return

        for file_path in skills_dir.iterdir():
            if file_path.is_file() and file_path.suffix == ".py" and not file_path.name.startswith("_"):
                module_name = f"app.skill.skills.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, SkillBase)
                            and attr is not SkillBase
                            and attr is not RedisSkill
                        ):
                            skill_instance = attr()
                            self._skills[skill_instance.name] = skill_instance
                            # 建立 tool -> skill 反向映射
                            for tool_name in skill_instance.tools:
                                self._tool_to_skill[tool_name] = skill_instance.name
                            logger.info("Skill 已加载（代码）: %s (tools=%s)", skill_instance.name, skill_instance.tools)
                except Exception as e:
                    logger.error("加载 Skill 模块失败: %s, 错误: %s", module_name, e)

    async def load_redis_skills(self) -> None:
        """从 Redis 加载手动配置的 Skill"""
        if self._redis_loaded:
            return

        try:
            from app.infrastructure.redis import RedisPool
            r = await RedisPool.get()
            keys = []
            async for key in r.scan_iter("digitalhuman:skill:*"):
                keys.append(key)

            for key in keys:
                entries = await r.hgetall(key)
                if not entries:
                    continue

                source = entries.get("source", "")
                # 只加载手动配置的 Skill（代码加载的已经在 load_skills 中处理）
                if source == "manual":
                    skill = RedisSkill(entries)
                    self._skills[skill.name] = skill
                    # 建立 tool -> skill 反向映射
                    for tool_name in skill.tools:
                        self._tool_to_skill[tool_name] = skill.name
                    logger.info("Skill 已加载（Redis）: %s (tools=%s)", skill.name, skill.tools)

            self._redis_loaded = True
        except Exception as e:
            logger.warning("加载 Redis Skill 失败: %s", e)

    def get_skill(self, name: str) -> SkillBase | None:
        return self._skills.get(name)

    def unload_skill(self, name: str) -> bool:
        """从内存卸载 Skill（删除时联动调用）

        Returns:
            True 如果成功卸载，False 如果 skill 不存在
        """
        skill = self._skills.pop(name, None)
        if skill is None:
            return False
        # 清理 tool -> skill 反向映射
        for tool_name in skill.tools:
            self._tool_to_skill.pop(tool_name, None)
        logger.info("Skill 已卸载: %s (tools=%s)", name, skill.tools)
        return True

    def list_skills(self) -> list[SkillBase]:
        return list(self._skills.values())

    def get_tools_for_skills(self, skill_ids: list[str]) -> list[dict]:
        """获取指定 Skills 声明的所有工具定义（OpenAI function calling 格式）"""
        tools = []
        for skill_id in skill_ids:
            skill = self._skills.get(skill_id)
            if skill:
                tools.extend(skill.get_tools_definitions())
        return tools

    def get_knowledge_collections(self, skill_ids: list[str]) -> list[str]:
        """获取指定 Skills 关联的 Milvus collection 列表"""
        collections = []
        for skill_id in skill_ids:
            skill = self._skills.get(skill_id)
            if skill and skill.knowledge_collection:
                collections.append(skill.knowledge_collection)
        return collections

    def get_skill_for_tool(self, tool_name: str) -> SkillBase | None:
        """根据工具名获取所属 Skill"""
        skill_name = self._tool_to_skill.get(tool_name)
        if skill_name:
            return self._skills.get(skill_name)
        return None


skill_loader = SkillLoader()
