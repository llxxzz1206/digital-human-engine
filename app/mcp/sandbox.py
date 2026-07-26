from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from app.config.settings import settings

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """工具执行异常"""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"工具 {tool_name} 执行失败: {message}")


class SandboxExecutor:
    """沙箱执行器：超时控制 + 权限检查 + 结构化错误"""

    # 高危权限列表
    _DANGEROUS_PERMISSIONS = {"filesystem:write", "network:outbound", "system:exec"}

    async def execute(
        self,
        handler: Callable,
        arguments: dict[str, Any],
        tool_name: str = "",
        permissions: list[str] | None = None,
        timeout: float | None = None,
        context: Any = None,
    ) -> Any:
        """在沙箱环境中执行工具

        Args:
            handler: 工具处理函数
            arguments: 工具参数
            tool_name: 工具名称（用于日志和错误信息）
            permissions: 工具权限列表
            timeout: 超时秒数，None 使用默认值
            context: SkillContext 运行时上下文

        Raises:
            ToolExecutionError: 执行超时或异常
        """
        # 1. 权限检查
        if permissions:
            dangerous = set(permissions) & self._DANGEROUS_PERMISSIONS
            if dangerous:
                logger.warning("工具 %s 请求高危权限: %s，已记录", tool_name, dangerous)

        # 2. 超时控制
        effective_timeout = timeout or settings.sandbox.default_timeout
        effective_timeout = min(effective_timeout, settings.sandbox.max_timeout)

        try:
            async with asyncio.timeout(effective_timeout):
                result = await handler(arguments, context)
            return result
        except TimeoutError:
            logger.error("工具执行超时: %s, timeout=%.1fs", tool_name, effective_timeout)
            raise ToolExecutionError(tool_name, f"执行超时 ({effective_timeout}s)")
        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error("工具执行异常: %s, error=%s", tool_name, e)
            raise ToolExecutionError(tool_name, str(e))


sandbox_executor = SandboxExecutor()
