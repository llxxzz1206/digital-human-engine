from __future__ import annotations

import logging
from datetime import datetime

from app.mcp.decorators import mcp_tool

logger = logging.getLogger(__name__)


@mcp_tool(
    name="echo",
    description="回显输入内容",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要回显的文本"},
        },
        "required": ["text"],
    },
)
async def echo_handler(arguments: dict, context: dict | None = None) -> dict:
    text = arguments.get("text", "")
    return {"text": text}


@mcp_tool(
    name="avatar_action",
    description="触发数字人切换视频状态",
    input_schema={
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "视频状态 (idle/talking/greeting/point_left/point_right/bow)",
                "enum": ["idle", "talking", "greeting", "point_left", "point_right", "bow"],
            },
        },
        "required": ["state"],
    },
)
async def avatar_action_handler(arguments: dict, context: dict | None = None) -> dict:
    state = arguments.get("state", "idle")
    return {"gesture": state}


@mcp_tool(
    name="current_time",
    description="获取当前时间",
    input_schema={
        "type": "object",
        "properties": {
            "format": {"type": "string", "description": "时间格式 (iso/timestamp)"},
        },
    },
)
async def current_time_handler(arguments: dict, context: dict | None = None) -> dict:
    fmt = arguments.get("format", "iso")
    now = datetime.now()
    if fmt == "timestamp":
        return {"time": int(now.timestamp() * 1000), "format": "timestamp"}
    return {"time": now.isoformat(), "format": "iso"}
