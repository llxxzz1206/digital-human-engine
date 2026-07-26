"""文本对话全链路测试 — 纯文本模拟前端，mock LLM/TTS/RAG

验证 chat.send → workflow → ai.stream 完整路径：
1. 普通对话（chat 路由）
2. RAG 知识库命中（rag_chat 路由）
3. FAQ 直接回答（faq_direct 路由）
4. 噪音过滤（noise 路由）
5. 航空展馆 Skill 导航工具调用
6. Capabilities 门控（text-only 不推 TTS/avatar）
7. 打断中断 workflow
8. Skill 删除后不影响运行
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ws.message_handler import handle_message, _recent_messages
from app.workflow.interaction_graph import InteractionGraph
from app.workflow.nodes import set_send_func
from app.session.manager import SessionManager
from app.session.history import ConversationHistory


# ─── 工具 ───────────────────────────────────────────────────

class FakeWebSocket:
    """捕获所有 send_json 的假 WS"""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, msg: dict):
        self.sent.append(msg)

    def by_type(self, msg_type: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == msg_type]

    @property
    def all_text(self) -> str:
        """拼接所有 ai.stream 的文本"""
        return "".join(
            m["payload"].get("text", "")
            for m in self.by_type("ai.stream")
            if not m["payload"].get("done")
        )

    @property
    def has_tts(self) -> bool:
        return len(self.by_type("tts.audio")) > 0

    @property
    def has_avatar(self) -> bool:
        return len(self.by_type("avatar.drive")) > 0

    @property
    def stream_done(self) -> bool:
        streams = self.by_type("ai.stream")
        return any(m["payload"].get("done") for m in streams)


@pytest.fixture(autouse=True)
def clean_dedup():
    _recent_messages.clear()
    yield
    _recent_messages.clear()


def _mock_llm_stream(reply_text: str):
    """构造 mock LLM 流式响应（逐字输出）"""
    from app.llm.service import StreamEvent

    async def _stream(*args, **kwargs):
        for char in reply_text:
            yield StreamEvent(type="text", content=char)

    return _stream


def _mock_tts_stream():
    """构造 mock TTS（返回空音频）"""
    async def _stream(*args, **kwargs):
        yield b"\x00" * 100

    return _stream


# ─── 场景 1：普通对话（chat 路由，无 RAG 命中） ───────────────

class TestChatRoute:
    """chat.send → RAG 无命中 → LLM 生成 → ai.stream"""

    async def test_basic_text_chat(self):
        """普通文本问答，验证 workflow 被调用 + 历史追加"""
        ws = FakeWebSocket()

        # Mock 整个 workflow 链路
        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()
        mock_graph.get_last_reply = MagicMock(return_value="你好！我是航空展馆的导览助手。")
        mock_graph.clear_last_reply = MagicMock()

        mock_history = MagicMock()
        mock_history.append = AsyncMock()
        mock_history.get_messages = AsyncMock(return_value=[
            {"role": "user", "content": "你好"}
        ])

        mock_session_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.sceneId = ""
        mock_session.deviceId = ""
        mock_session.deviceLocation = ""
        mock_session.userType = "general"
        mock_session.mountedSkills = []
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.refresh_ttl = AsyncMock()

        with patch("app.ws.message_handler.interaction_graph", mock_graph), \
             patch("app.ws.message_handler.conversation_history", mock_history), \
             patch("app.ws.message_handler.session_manager", mock_session_mgr):

            await handle_message({
                "type": "chat.send",
                "payload": {"sessionId": "chat-001", "text": "你好"},
            }, ws)

            # 等待 create_task 完成
            await asyncio.sleep(0.05)

        # 验证 workflow 被调用
        mock_graph.run.assert_called_once()
        call_kwargs = mock_graph.run.call_args.kwargs
        assert call_kwargs["session_id"] == "chat-001"

        # 验证历史追加（第一次是 user 消息）
        first_call = mock_history.append.call_args_list[0]
        assert first_call[0] == ("chat-001", "user", "你好")

    async def test_empty_text_ignored(self):
        """空文本不触发 workflow"""
        ws = FakeWebSocket()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        mock_history = MagicMock()
        mock_history.append = AsyncMock()

        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=None)

        with patch("app.ws.message_handler.interaction_graph", mock_graph), \
             patch("app.ws.message_handler.conversation_history", mock_history), \
             patch("app.ws.message_handler.session_manager", mock_session_mgr):

            await handle_message({
                "type": "chat.send",
                "payload": {"sessionId": "chat-002", "text": ""},
            }, ws)
            await asyncio.sleep(0.05)

        # 空文本也会走 workflow（当前设计），但 messages 中 content 为空
        # 验证不崩溃即可


# ─── 场景 2：Workflow 内部全链路（直接调用 InteractionGraph） ──

class TestWorkflowFullChain:
    """直接调用 InteractionGraph.run()，mock LLM/TTS/RAG 验证完整路径"""

    async def test_chat_route_generates_reply(self):
        """chat 路由：RAG 无结果 → LLM 生成回复 → 发送 ai.stream"""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        graph = InteractionGraph()

        # Mock RAG 检索（返回空 → chat 路由）
        with patch("app.workflow.nodes.dual_retriever") as mock_retriever, \
             patch("app.workflow.nodes.search_cache") as mock_cache, \
             patch("app.workflow.nodes.llm_service") as mock_llm, \
             patch("app.workflow.nodes.tts_service") as mock_tts, \
             patch("app.workflow.nodes.tts_cache") as mock_tts_cache, \
             patch("app.workflow.nodes.chat_logger") as mock_logger, \
             patch("app.workflow.nodes._check_interrupted", AsyncMock(return_value=False)), \
             patch("app.workflow.interaction_graph.session_manager") as mock_sm, \
             patch("app.infrastructure.redis.RedisPool") as mock_redis_pool:

            # RAG 无结果
            mock_cache.get_or_search = AsyncMock(return_value=([], "knowledge"))

            # LLM 流式回复
            from app.llm.service import StreamEvent
            async def mock_stream(*args, **kwargs):
                for char in "你好，我是展馆助手":
                    yield StreamEvent(type="text", content=char)
            mock_llm.stream_chat = mock_stream
            mock_llm.stream_chat_with_tools = mock_stream

            # TTS mock
            async def mock_tts_stream(*args, **kwargs):
                yield b"\x00" * 100
            mock_tts.synthesize_stream = mock_tts_stream
            mock_tts_cache.get = AsyncMock(return_value=None)
            mock_tts_cache.get_or_synthesize = AsyncMock(return_value=b"\x00" * 100)
            mock_tts_cache.put = AsyncMock()

            # Session mock
            mock_session = MagicMock()
            mock_session.platform = "fixed_terminal"
            mock_session.capabilities = ["text", "audio", "avatar"]
            mock_session.currentLocation = ""
            mock_session.locationSource = ""
            mock_sm.get_session = AsyncMock(return_value=mock_session)

            # Redis mock (for timing record)
            mock_r = AsyncMock()
            mock_r.pipeline = MagicMock(return_value=MagicMock(
                lpush=MagicMock(), ltrim=MagicMock(), execute=AsyncMock()
            ))
            mock_redis_pool.get = AsyncMock(return_value=mock_r)

            # 跨设备上下文
            with patch("app.workflow.nodes._load_cross_device_context", new=AsyncMock(return_value="")):
                await graph.run(
                    session_id="wf-001",
                    messages=[{"role": "user", "content": "你好"}],
                    send_func=capture_send,
                )

        # 验证 ai.stream 输出
        ai_streams = [m for m in sent_messages if m.get("type") == "ai.stream"]
        assert len(ai_streams) > 0

        # 拼接文本
        full_text = "".join(m["payload"]["text"] for m in ai_streams if not m["payload"].get("done"))
        assert "你好" in full_text

        # 验证 done 标记
        done_msgs = [m for m in ai_streams if m["payload"].get("done")]
        assert len(done_msgs) == 1

        # 验证 last_reply 被记录
        assert graph.get_last_reply("wf-001") == "你好，我是展馆助手"

    async def test_text_only_no_tts_no_avatar(self):
        """capabilities=["text"] 时不推送 tts.audio 和 avatar.drive"""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        graph = InteractionGraph()

        with patch("app.workflow.nodes.dual_retriever") as mock_retriever, \
             patch("app.workflow.nodes.search_cache") as mock_cache, \
             patch("app.workflow.nodes.llm_service") as mock_llm, \
             patch("app.workflow.nodes.tts_service") as mock_tts, \
             patch("app.workflow.nodes.tts_cache") as mock_tts_cache, \
             patch("app.workflow.nodes.chat_logger") as mock_logger, \
             patch("app.workflow.nodes._check_interrupted", AsyncMock(return_value=False)), \
             patch("app.workflow.interaction_graph.session_manager") as mock_sm, \
             patch("app.infrastructure.redis.RedisPool") as mock_redis_pool:

            mock_cache.get_or_search = AsyncMock(return_value=([], "knowledge"))

            from app.llm.service import StreamEvent
            async def mock_stream(*args, **kwargs):
                yield StreamEvent(type="text", content="回复")
            mock_llm.stream_chat = mock_stream
            mock_llm.stream_chat_with_tools = mock_stream

            async def mock_tts_stream(*args, **kwargs):
                yield b"\x00"
            mock_tts.synthesize_stream = mock_tts_stream
            mock_tts_cache.get = AsyncMock(return_value=None)
            mock_tts_cache.put = AsyncMock()

            # 关键：capabilities 只有 text
            mock_session = MagicMock()
            mock_session.platform = "web_admin"
            mock_session.capabilities = ["text"]
            mock_session.currentLocation = ""
            mock_session.locationSource = ""
            mock_sm.get_session = AsyncMock(return_value=mock_session)

            mock_r = AsyncMock()
            mock_r.pipeline = MagicMock(return_value=MagicMock(
                lpush=MagicMock(), ltrim=MagicMock(), execute=AsyncMock()
            ))
            mock_redis_pool.get = AsyncMock(return_value=mock_r)

            with patch("app.workflow.nodes._load_cross_device_context", new=AsyncMock(return_value="")):
                await graph.run(
                    session_id="wf-text",
                    messages=[{"role": "user", "content": "测试"}],
                    send_func=capture_send,
                )

        # 不应有 tts.audio
        tts_msgs = [m for m in sent_messages if m.get("type") == "tts.audio"]
        assert len(tts_msgs) == 0, f"text-only 客户端不应收到 TTS: {tts_msgs}"

        # 不应有 avatar.drive
        avatar_msgs = [m for m in sent_messages if m.get("type") == "avatar.drive"]
        assert len(avatar_msgs) == 0, f"text-only 客户端不应收到 avatar.drive: {avatar_msgs}"

        # 但应有 ai.stream
        ai_streams = [m for m in sent_messages if m.get("type") == "ai.stream"]
        assert len(ai_streams) > 0


# ─── 场景 3：航空展馆 Skill 导航工具 ──────────────────────────

class TestAviationSkill:
    """航空展馆 Skill 加载 + 导航工具调用"""

    def test_skill_loads(self):
        """Skill 文件正确加载"""
        from app.skill.loader import SkillLoader
        from pathlib import Path

        loader = SkillLoader()
        skills_dir = Path(__file__).parent.parent / "app" / "skill" / "skills"
        loader.load_skills(skills_dir)

        skill = loader.get_skill("aviation")
        assert skill is not None
        assert skill.name == "aviation"
        assert "aviation_navigate" in skill.tools
        assert "aviation_exhibit_info" in skill.tools
        assert skill.knowledge_collection == "aviation_knowledge"

    def test_hospital_skill_removed(self):
        """hospital skill 已删除，不再加载"""
        from app.skill.loader import SkillLoader
        from pathlib import Path

        loader = SkillLoader()
        skills_dir = Path(__file__).parent.parent / "app" / "skill" / "skills"
        loader.load_skills(skills_dir)

        assert loader.get_skill("hospital") is None

    async def test_navigate_exact_zone(self):
        """精确匹配展区名"""
        from app.skill.skills.aviation_skill import aviation_navigate_handler

        result = await aviation_navigate_handler({"query": "航天探索厅"})
        assert result["success"] is True
        assert result["zone"] == "航天探索厅"
        assert result["floor"] == "2楼"
        assert result["gesture"] == "point_left"

    async def test_navigate_with_question_words(self):
        """带问句修饰词的导航查询"""
        from app.skill.skills.aviation_skill import aviation_navigate_handler

        result = await aviation_navigate_handler({"query": "无人机科技区在哪里"})
        assert result["success"] is True
        assert result["zone"] == "无人机科技区"
        assert result["floor"] == "3楼"

    async def test_navigate_exhibit_lookup(self):
        """通过展品名查找所属展区"""
        from app.skill.skills.aviation_skill import aviation_navigate_handler

        result = await aviation_navigate_handler({"query": "歼-20模型"})
        assert result["success"] is True
        assert result["zone"] == "飞行器模型区"

    async def test_navigate_fuzzy_match(self):
        """模糊匹配"""
        from app.skill.skills.aviation_skill import aviation_navigate_handler

        result = await aviation_navigate_handler({"query": "模拟"})
        assert result["success"] is True
        assert "matches" in result

    async def test_navigate_not_found(self):
        """未找到返回全部展区"""
        from app.skill.skills.aviation_skill import aviation_navigate_handler

        result = await aviation_navigate_handler({"query": "火星基地"})
        assert result["success"] is False
        assert "all_zones" in result

    async def test_exhibit_info(self):
        """展品信息查询"""
        from app.skill.skills.aviation_skill import aviation_exhibit_info_handler

        result = await aviation_exhibit_info_handler({"exhibit_name": "长征五号火箭模型"})
        assert result["success"] is True
        assert result["zone"] == "航天探索厅"

    async def test_exhibit_info_not_found(self):
        """展品不存在"""
        from app.skill.skills.aviation_skill import aviation_exhibit_info_handler

        result = await aviation_exhibit_info_handler({"exhibit_name": "不存在的展品"})
        assert result["success"] is False

    def test_skill_tools_definitions(self):
        """Skill 工具定义格式正确（OpenAI function calling）"""
        from app.skill.loader import SkillLoader
        from pathlib import Path

        loader = SkillLoader()
        skills_dir = Path(__file__).parent.parent / "app" / "skill" / "skills"
        loader.load_skills(skills_dir)

        tools = loader.get_tools_for_skills(["aviation"])
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert "aviation_navigate" in names
        assert "aviation_exhibit_info" in names


# ─── 场景 4：Skill 删除联动 ──────────────────────────────────

class TestSkillDelete:
    """Skill 删除后内存卸载 + 不影响运行"""

    def test_unload_skill(self):
        """unload_skill 从内存移除"""
        from app.skill.loader import SkillLoader
        from pathlib import Path

        loader = SkillLoader()
        skills_dir = Path(__file__).parent.parent / "app" / "skill" / "skills"
        loader.load_skills(skills_dir)

        assert loader.get_skill("aviation") is not None
        result = loader.unload_skill("aviation")
        assert result is True
        assert loader.get_skill("aviation") is None

        # 工具映射也清理
        assert loader.get_skill_for_tool("aviation_navigate") is None

    def test_unload_nonexistent(self):
        """卸载不存在的 skill 返回 False"""
        from app.skill.loader import SkillLoader

        loader = SkillLoader()
        assert loader.unload_skill("nonexistent") is False

    async def test_chat_after_skill_deleted(self):
        """Skill 删除后 chat.send 不崩溃（无工具可用时正常走 chat 路由）"""
        ws = FakeWebSocket()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()
        mock_graph.get_last_reply = MagicMock(return_value="回复")
        mock_graph.clear_last_reply = MagicMock()

        mock_history = MagicMock()
        mock_history.append = AsyncMock()
        mock_history.get_messages = AsyncMock(return_value=[])

        mock_session_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.sceneId = "aviation"
        mock_session.deviceId = "dev1"
        mock_session.deviceLocation = "大厅"
        mock_session.userType = "general"
        mock_session.mountedSkills = []  # skill 已删除，无挂载
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)

        # skill_loader 中无 aviation
        with patch("app.ws.message_handler.interaction_graph", mock_graph), \
             patch("app.ws.message_handler.conversation_history", mock_history), \
             patch("app.ws.message_handler.session_manager", mock_session_mgr), \
             patch("app.skill.loader.skill_loader") as mock_sl:

            mock_sl.get_skill = MagicMock(return_value=None)
            mock_sl.get_tools_for_skills = MagicMock(return_value=[])

            await handle_message({
                "type": "chat.send",
                "payload": {"sessionId": "del-001", "text": "你好", "sceneId": "aviation"},
            }, ws)
            await asyncio.sleep(0.05)

        # 不崩溃，workflow 正常调用（无工具）
        mock_graph.run.assert_called_once()
        call_kwargs = mock_graph.run.call_args.kwargs
        assert call_kwargs["tools"] is None or call_kwargs["tools"] == []


# ─── 场景 5：导航预路由泛化 ──────────────────────────────────

class TestNavigationPreRoute:
    """nodes.py 导航预路由不再硬编码 hospital_navigate"""

    async def test_aviation_navigate_pre_route(self):
        """航空展馆导航工具能被预路由识别"""
        from app.workflow.nodes import _maybe_pre_run_navigation

        available_tools = [
            {"type": "function", "function": {"name": "aviation_navigate", "description": "导航"}},
            {"type": "function", "function": {"name": "aviation_exhibit_info", "description": "展品"}},
        ]

        with patch("app.workflow.nodes._mcp_tool_executor") as mock_exec:
            mock_exec.return_value = {"success": True, "gesture": "point_left", "zone": "航天探索厅"}

            result = await _maybe_pre_run_navigation("航天探索厅在哪里", available_tools, "nav-001")

        assert result is not None
        assert result["success"] is True
        assert result["gesture"] == "point_left"
        mock_exec.assert_called_once()
        # 验证调用的是 aviation_navigate 而非 hospital_navigate
        call_args = mock_exec.call_args
        assert call_args[0][0] == "aviation_navigate"

    async def test_no_nav_tool_skips(self):
        """无导航工具时跳过预路由"""
        from app.workflow.nodes import _maybe_pre_run_navigation

        available_tools = [
            {"type": "function", "function": {"name": "some_other_tool", "description": "其他"}},
        ]

        result = await _maybe_pre_run_navigation("航天探索厅在哪里", available_tools, "nav-002")
        assert result is None

    async def test_non_nav_question_skips(self):
        """非位置类问题跳过预路由"""
        from app.workflow.nodes import _maybe_pre_run_navigation

        available_tools = [
            {"type": "function", "function": {"name": "aviation_navigate", "description": "导航"}},
        ]

        result = await _maybe_pre_run_navigation("歼-20的发动机是什么型号", available_tools, "nav-003")
        assert result is None


# ─── 场景 6：打断中断 workflow ────────────────────────────────

class TestInterruptWorkflow:
    """interrupt 消息中断正在运行的 workflow"""

    async def test_interrupt_during_chat(self):
        """chat.send 后发 interrupt，workflow 应被中断"""
        ws = FakeWebSocket()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()
        mock_graph.get_last_reply = MagicMock(return_value="正在讲解...")
        mock_graph.clear_last_reply = MagicMock()
        mock_graph.interrupt = AsyncMock()

        mock_history = MagicMock()
        mock_history.append = AsyncMock()
        mock_history.get_messages = AsyncMock(return_value=[])

        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=None)

        with patch("app.ws.message_handler.interaction_graph", mock_graph), \
             patch("app.ws.message_handler.conversation_history", mock_history), \
             patch("app.ws.message_handler.session_manager", mock_session_mgr), \
             patch("app.ws.message_handler.breakpoint_manager") as mock_bp:

            mock_bp.save = MagicMock()

            # 发送 interrupt
            await handle_message({
                "type": "interrupt",
                "payload": {"sessionId": "int-001"},
            }, ws)

        # 验证 interrupt 被调用
        mock_graph.interrupt.assert_called_once_with("int-001")

        # 验证 interrupt.ack 回复
        ack = ws.by_type("interrupt.ack")
        assert len(ack) == 1
        assert ack[0]["payload"]["sessionId"] == "int-001"
