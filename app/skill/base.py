from __future__ import annotations

from abc import ABC, abstractmethod

from app.mcp.registry import tool_registry


class SkillBase(ABC):
    """Skill 抽象基类"""

    name: str = ""
    description: str = ""
    tools: list[str] = []  # 工具名称列表
    knowledge_collection: str = ""  # Milvus collection 名
    system_prompt: str = ""  # Skill 专属系统提示词

    @abstractmethod
    async def on_mount(self, session_id: str) -> None:
        """挂载时调用：注册工具、加载知识库"""
        ...

    @abstractmethod
    async def on_unmount(self, session_id: str) -> None:
        """卸载时调用：清理资源"""
        ...

    def get_tools_definitions(self) -> list[dict]:
        """返回此 Skill 声明的工具定义（从 ToolRegistry 提取）"""
        definitions = []
        for tool_name in self.tools:
            tool = tool_registry.get_tool(tool_name)
            if tool:
                definitions.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                })
        return definitions

    def get_system_prompt(self) -> str:
        """返回 Skill 专属系统提示词"""
        return self.system_prompt
