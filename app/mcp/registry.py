from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: Callable | None = None
    permissions: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
        handler: Callable | None = None,
        permissions: list[str] | None = None,
    ) -> None:
        tool = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema or {},
            handler=handler,
            permissions=permissions or [],
        )
        self._tools[name] = tool
        logger.info("MCP 工具已注册: %s", name)

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute(self, name: str, arguments: dict[str, Any], context: dict[str, Any] | None = None) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"工具不存在: {name}")
        if tool.handler is None:
            raise ValueError(f"工具无执行处理器: {name}")
        logger.info("执行 MCP 工具: %s, 参数: %s", name, arguments)
        result = await tool.handler(arguments, context)
        return result


tool_registry = ToolRegistry()
