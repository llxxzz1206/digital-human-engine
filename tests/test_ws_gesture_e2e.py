"""真实契约端到端回归：前端只发 sceneId（不发 skillIds），导航手势必须触发

回归背景：此前测试手动带 skill_ids=["aviation"]，掩盖了真实前端不发
skillIds 导致工具列表为空、手势永远不触发的问题（修复在 message_handler.py
的 scene→skill 解析）。若该解析被删除，本测试必须红。

前置：Python AI Engine 运行在 localhost:8000（未运行时自动跳过）。
运行：uv run python -m pytest tests/test_ws_gesture_e2e.py -v
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
import uuid

import pytest
import websockets

URI = "ws://localhost:8000/ws"


def _server_alive() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_alive(), reason="AI Engine 未运行 (localhost:8000)"
)


async def _ask(text: str) -> tuple[str, list[str]]:
    """按前端真实契约发问（只带 sceneId），返回 (回复文本, avatar.drive 状态序列)"""
    session_id = f"pytest-{uuid.uuid4().hex[:8]}"
    async with websockets.connect(URI, open_timeout=10) as ws:
        await ws.send(json.dumps({
            "type": "session.create",
            "payload": {
                "sessionId": session_id,
                "sceneId": "aviation",  # 真实前端只发 sceneId，不发 skillIds
                "deviceId": "pytest-device",
                "deviceLocation": "1楼大厅",
            },
        }))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert resp["type"] == "session.created", resp

        await ws.send(json.dumps({
            "type": "chat.send",
            "payload": {"sessionId": session_id, "text": text},
        }))

        reply, gestures = "", []
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                msg_type = msg.get("type")
                payload = msg.get("payload", {})
                if msg_type == "ai.stream":
                    reply += payload.get("text", "") or payload.get("delta", "")
                elif msg_type == "avatar.drive":
                    gestures.append(payload.get("state"))
        except asyncio.TimeoutError:
            pass
        return reply, gestures


@pytest.mark.parametrize("question, want_zone, want_gesture", [
    ("航空历史厅在哪", "1F", "point_right"),   # 右翼
    ("航天探索厅在哪", "2F", "point_left"),    # 左翼
])
async def test_navigation_gesture_via_scene_contract(question, want_zone, want_gesture):
    reply, gestures = await _ask(question)
    assert want_zone in reply, (
        f"回复未提及 {want_zone}: {reply[:80]}"
    )
    assert want_gesture in gestures, f"未触发 {want_gesture}: {gestures}"
