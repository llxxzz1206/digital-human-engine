"""WS 消息处理主链路单元测试（项4）

用 mock 锁住核心链路（无需真实服务）：
- session.create → session.created 回显场景/设备信息
- chat.send：只发 sceneId 不带 skillIds 时，后端必须自行解析技能且工具非空
  （项2 修复的回归点：解析丢失 → 工具列表空 → 导航手势永远不触发）
- chat.send：同 session 同文本 3 秒内去重，workflow 只跑一次
- interrupt → interrupt.ack
- 未知消息类型静默忽略
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ws import message_handler
from app.ws.message_handler import handle_message


class FakeWebSocket:
    """捕获 send_json 的假 WS"""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, msg: dict) -> None:
        self.sent.append(msg)

    def by_type(self, msg_type: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == msg_type]


@pytest.fixture
def ws() -> FakeWebSocket:
    return FakeWebSocket()


@pytest.fixture(autouse=True)
def _clean_dedup_window():
    message_handler._recent_messages.clear()
    yield
    message_handler._recent_messages.clear()


async def _drain_background_tasks(rounds: int = 20) -> None:
    """chat.send 用 create_task 异步跑 workflow，yield 若干轮让 mock 链路跑完"""
    for _ in range(rounds):
        await asyncio.sleep(0)


def _patch_chain():
    """把 workflow/历史/会话依赖换成 mock，返回 (graph_mock, history_mock) 的上下文管理器"""
    graph = MagicMock()
    graph.run = AsyncMock()
    graph.get_last_reply = MagicMock(return_value=None)
    graph.interrupt = AsyncMock()

    history = MagicMock()
    history.append = AsyncMock()
    history.get_messages = AsyncMock(return_value=[])

    return graph, history


async def test_session_create_echoes_scene_and_device(ws: FakeWebSocket):
    session = SimpleNamespace(
        sessionId="s1", userId="u1", avatarId="a1", mountedSkills=[],
        sceneId="aviation", deviceId="d1", deviceLocation="一楼大厅",
        userType="general",
    )
    with patch.object(message_handler.session_manager, "create_session",
                      new=AsyncMock(return_value=session)):
        await handle_message({
            "type": "session.create",
            "payload": {"sessionId": "s1", "sceneId": "aviation",
                        "deviceId": "d1", "deviceLocation": "一楼大厅"},
        }, ws)

    created = ws.by_type("session.created")
    assert len(created) == 1
    payload = created[0]["payload"]
    assert payload["sessionId"] == "s1"
    assert payload["sceneId"] == "aviation"
    assert payload["deviceId"] == "d1"
    assert payload["deviceLocation"] == "一楼大厅"


async def test_chat_send_resolves_skill_from_scene_id_only(ws: FakeWebSocket):
    """回归锁定：前端只发 sceneId（真实契约），后端必须解析出技能与工具"""
    from app.skill.loader import skill_loader

    if not skill_loader.get_skill("aviation"):
        skills_dir = Path(message_handler.__file__).parent.parent / "skill" / "skills"
        skill_loader.load_skills(skills_dir)

    graph, history = _patch_chain()
    with patch.object(message_handler, "interaction_graph", graph), \
         patch.object(message_handler, "conversation_history", history), \
         patch.object(message_handler.session_manager, "get_session",
                      new=AsyncMock(return_value=None)), \
         patch.object(message_handler.session_manager, "refresh_ttl", new=AsyncMock()):
        await handle_message({
            "type": "chat.send",
            "payload": {"sessionId": "s-nav", "text": "航空历史厅在哪", "sceneId": "aviation"},
        }, ws)
        await _drain_background_tasks()

    assert graph.run.await_count == 1
    kwargs = graph.run.await_args.kwargs
    assert kwargs["skill_ids"] == ["aviation"], "sceneId 未解析成技能，工具列表将为空"
    assert kwargs["tools"], "工具列表为空，导航手势无法触发"
    tool_names = {t["function"]["name"] for t in kwargs["tools"]}
    assert "aviation_navigate" in tool_names


async def test_chat_send_explicit_skill_ids_not_overridden(ws: FakeWebSocket):
    graph, history = _patch_chain()
    with patch.object(message_handler, "interaction_graph", graph), \
         patch.object(message_handler, "conversation_history", history), \
         patch.object(message_handler.session_manager, "get_session",
                      new=AsyncMock(return_value=None)), \
         patch.object(message_handler.session_manager, "refresh_ttl", new=AsyncMock()):
        await handle_message({
            "type": "chat.send",
            "payload": {"sessionId": "s2", "text": "你好",
                        "sceneId": "aviation", "skillIds": ["aviation"]},
        }, ws)
        await _drain_background_tasks()

    assert graph.run.await_args.kwargs["skill_ids"] == ["aviation"]


async def test_chat_send_dedup_within_window(ws: FakeWebSocket):
    graph, history = _patch_chain()
    with patch.object(message_handler, "interaction_graph", graph), \
         patch.object(message_handler, "conversation_history", history), \
         patch.object(message_handler.session_manager, "get_session",
                      new=AsyncMock(return_value=None)), \
         patch.object(message_handler.session_manager, "refresh_ttl", new=AsyncMock()):
        msg = {"type": "chat.send",
               "payload": {"sessionId": "s-dup", "text": "你好", "skillIds": []}}
        await handle_message(msg, ws)
        await handle_message(msg, ws)  # 3 秒窗口内的重复消息
        await _drain_background_tasks()

    assert graph.run.await_count == 1, "重复消息不应再次触发 workflow"


async def test_interrupt_sends_ack(ws: FakeWebSocket):
    graph, _ = _patch_chain()
    with patch.object(message_handler, "interaction_graph", graph):
        await handle_message({"type": "interrupt", "payload": {"sessionId": "s1"}}, ws)

    graph.interrupt.assert_awaited_once_with("s1")
    ack = ws.by_type("interrupt.ack")
    assert len(ack) == 1
    assert ack[0]["payload"]["sessionId"] == "s1"


async def test_unknown_message_type_is_ignored(ws: FakeWebSocket):
    await handle_message({"type": "no.such.type", "payload": {}}, ws)
    assert ws.sent == []
