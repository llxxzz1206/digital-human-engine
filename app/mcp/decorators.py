from __future__ import annotations

from typing import Any, Callable

from app.mcp.registry import tool_registry


def mcp_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any] | None = None,
    permissions: list[str] | None = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        tool_registry.register(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=func,
            permissions=permissions,
        )
        return func

    return decorator
