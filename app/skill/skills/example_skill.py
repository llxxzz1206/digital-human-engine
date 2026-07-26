from __future__ import annotations

import logging

from app.mcp.decorators import mcp_tool
from app.skill.base import SkillBase

logger = logging.getLogger(__name__)


class ExampleSkill(SkillBase):
    name = "example"
    description = "示例技能，提供基础回显工具"
    tools = ["example_echo"]
    knowledge_collection = "example_knowledge"
    system_prompt = "你是一个示例数字人助手。你可以使用 example_echo 工具回显用户输入的内容。"

    async def on_mount(self, session_id: str) -> None:
        logger.info("ExampleSkill 挂载到会话: %s", session_id)

    async def on_unmount(self, session_id: str) -> None:
        logger.info("ExampleSkill 从会话卸载: %s", session_id)


@mcp_tool(
    name="example_echo",
    description="示例 Skill 的回显工具，将输入原样返回",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要回显的文本"},
        },
        "required": ["text"],
    },
)
async def example_echo_handler(arguments: dict, context: dict | None = None) -> dict:
    text = arguments.get("text", "")
    return {"echo": text}
