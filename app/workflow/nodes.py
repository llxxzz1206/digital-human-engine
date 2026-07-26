from __future__ import annotations

import json
import asyncio
import base64
import logging
import time
import contextvars
from typing import Any, Awaitable, Callable

from app.llm.service import llm_service, StreamEvent
from app.rag.retriever import dual_retriever
from app.rag.reranker import reranker
from app.rag.tts_cache import tts_cache
from app.rag.search_cache import search_cache
from app.mcp.registry import tool_registry
from app.mcp.sandbox import sandbox_executor
from app.avatar.driver import AvatarDriver, avatar_driver
from app.voice.tts_service import tts_service
from app.services.chat_logger import chat_logger
from app.workflow.state import WorkflowState
from app.config.settings import settings

logger = logging.getLogger(__name__)

# 使用 ContextVar 替代全局变量，确保并发 session 各自独立
_current_send_func: contextvars.ContextVar[Callable[[dict], Awaitable[None]] | None] = contextvars.ContextVar(
    "_current_send_func", default=None
)

# Avatar 驱动器：per-session 缓存（从形象配置构造）
_session_drivers: dict[str, AvatarDriver] = {}


def get_session_driver(session_id: str, avatar_config: dict | None = None) -> AvatarDriver:
    """获取 session 对应的 AvatarDriver（有配置则构造，否则用默认）"""
    if session_id in _session_drivers:
        return _session_drivers[session_id]
    if avatar_config:
        driver = AvatarDriver.from_config(avatar_config)
    else:
        driver = avatar_driver  # 默认实例
    _session_drivers[session_id] = driver
    return driver


def clear_session_driver(session_id: str) -> None:
    """session 结束时清理"""
    _session_drivers.pop(session_id, None)

# 句子结束符
_SENTENCE_ENDS = set("。？！；\n!?;")


def set_send_func(func: Callable[[dict], Awaitable[None]] | None) -> None:
    """设置当前 send_func（由 InteractionGraph.run() 调用）"""
    _current_send_func.set(func)


async def _send(msg: dict) -> None:
    """发送消息到客户端"""
    func = _current_send_func.get()
    if func:
        try:
            await func(msg)
        except Exception as e:
            logger.error("发送消息失败: %s", e)


# 工具执行中捕获的 gesture（供 avatar_driver_node 使用）
_last_gesture: dict[str, str] = {}  # session_id -> gesture


async def _mcp_tool_executor(tool_name: str, arguments: dict, session_id: str = "", context: Any = None) -> Any:
    """MCP 工具执行器：桥接 LLM tool_call 到 MCP registry

    由 llm_service.stream_chat_with_tools() 的 agentic loop 调用。
    """
    tool = tool_registry.get_tool(tool_name)
    if tool is None or tool.handler is None:
        return {"error": f"工具 {tool_name} 不存在或无处理器"}

    try:
        result = await sandbox_executor.execute(
            handler=tool.handler,
            arguments=arguments,
            tool_name=tool_name,
            permissions=tool.permissions,
            context=context,
        )
        logger.info("MCP 工具执行成功: %s(%s)", tool_name, arguments)
        # 捕获 gesture 提示
        if isinstance(result, dict) and result.get("gesture") and session_id:
            _last_gesture[session_id] = result["gesture"]
        return result
    except Exception as e:
        logger.error("MCP 工具执行失败: %s, error=%s", tool_name, e)
        return {"error": str(e)}


# 导航意图关键词（命中即认为是位置类问题）
_NAV_KEYWORDS = ("在哪", "在哪里", "怎么走", "几楼", "几层", "位置", "在什么", "哪个楼", "怎么去", "哪里", "哪个区", "哪个展")


async def _maybe_pre_run_navigation(user_input: str, available_tools: list[dict], session_id: str) -> dict | None:
    """检测导航意图并预执行导航工具（保证指引手势触发）

    若用户输入是位置类问题且可用工具中存在名称含 'navigate' 的工具，直接执行。
    工具结果中的 gesture 会被 _mcp_tool_executor 捕获，供 avatar_driver_node 使用。
    支持任意 Skill 提供的导航工具（hospital_navigate / aviation_navigate 等）。
    """
    if not user_input or not any(kw in user_input for kw in _NAV_KEYWORDS):
        return None

    # 查找可用工具中名称含 'navigate' 的工具
    nav_tool_name = None
    for t in available_tools:
        name = t.get("function", {}).get("name", "")
        if "navigate" in name:
            nav_tool_name = name
            break

    if not nav_tool_name:
        return None

    result = await _mcp_tool_executor(nav_tool_name, {"query": user_input}, session_id)
    if isinstance(result, dict) and result.get("success"):
        logger.info("导航预路由命中: tool=%s, input=%s, gesture=%s", nav_tool_name, user_input[:30], result.get("gesture"))
        return result
    return None


def get_all_available_tools(skill_tools: list[dict] | None = None) -> list[dict]:
    """合并 Skill 工具 + 内置 MCP 工具，返回 OpenAI function calling 格式列表"""
    tools: list[dict] = list(skill_tools) if skill_tools else []

    # 追加已注册的内置 MCP 工具（避免重名）
    existing_names = {t.get("function", {}).get("name", "") for t in tools}
    for mcp_tool in tool_registry.list_tools():
        if mcp_tool.name not in existing_names and mcp_tool.handler is not None:
            tools.append({
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "description": mcp_tool.description,
                    "parameters": mcp_tool.input_schema or {"type": "object", "properties": {}},
                },
            })

    return tools


def _is_sentence_end(text: str) -> bool:
    """检测文本是否以句子结束符结尾"""
    return bool(text) and text.rstrip()[-1:] in _SENTENCE_ENDS


async def _check_interrupted(session_id: str) -> bool:
    """检查 session 是否被中断"""
    try:
        from app.infrastructure.redis import RedisPool
        redis = await RedisPool.get()
        key = f"digitalhuman:interrupt:{session_id}"
        val = await redis.get(key)
        if val:
            # 清除标志
            await redis.delete(key)
            return True
    except Exception:
        pass
    return False


async def _summarize_history(old_messages: list[dict]) -> str:
    """将超出窗口的旧消息压缩为摘要（非流式 LLM 调用）

    如果消息太少（<=4条）则直接拼接文本，不调 LLM。
    """
    if not old_messages:
        return ""

    # 提取文本内容
    lines = []
    for msg in old_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            prefix = "用户" if role == "user" else "助手"
            lines.append(f"{prefix}: {content[:100]}")

    if not lines:
        return ""

    # 消息太少，直接拼接
    if len(lines) <= 4:
        return "；".join(lines)

    # 用 LLM 生成摘要
    try:
        summary_prompt = (
            "请将以下对话历史压缩为一段简短摘要（不超过100字），"
            "保留关键信息（用户问了什么、得到了什么答案）：\n\n"
            + "\n".join(lines[-20:])  # 最多取最近20条旧消息
        )
        result = await llm_service.chat([
            {"role": "system", "content": "你是摘要助手，只输出摘要内容，不加任何前缀。"},
            {"role": "user", "content": summary_prompt},
        ])
        return result.strip() if result else ""
    except Exception as e:
        logger.error("历史摘要生成失败: %s", e)
        # 降级：取最后几条拼接
        return "；".join(lines[-4:])


async def _background_summarize(session_id: str, old_messages: list[dict]) -> None:
    """后台生成摘要并缓存，不阻塞回复路径"""
    try:
        summary = await _summarize_history(old_messages)
        if summary:
            from app.session.history import conversation_history
            await conversation_history.set_summary(session_id, summary, len(old_messages))
            logger.debug("后台摘要已缓存: session=%s, len=%d", session_id, len(summary))
    except Exception as e:
        logger.warning("后台摘要刷新失败: %s", e)


async def rag_retriever_node(state: WorkflowState) -> dict:
    """RAG 检索 + 重排节点：先查 FAQ，再查知识库

    双集合检索（支持多场景多设备分层）：
    1. 先查 faq_* 集合（LLM 润色后的回答），高分命中可直接输出
    2. FAQ 未命中则查 skill_* 集合（结构化知识），永远走 rag_chat
    3. 均无结果走 chat

    有 scene_id/device_id 时按层级搜索：
    设备级 FAQ → 场景级 FAQ → 设备级知识库 → 场景级知识库

    同时读取跨设备上下文，供后续节点注入 LLM prompt。
    """
    user_input = state.get("user_input", "")
    skill_ids = state.get("skill_ids", [])
    scene_id = state.get("scene_id", "")
    device_id = state.get("device_id", "")
    platform = state.get("platform", "fixed_terminal")
    available_tools = state.get("available_tools", [])
    session_id = state.get("session_id", "")

    # 分阶段计时：合并 state 已有 timings（如 ASR），追加 RAG 段
    timings = dict(state.get("timings", {}))
    t_node_start = time.monotonic()

    # 导航意图预路由：位置类问题先执行 hospital_navigate，确定指引手势。
    # 放在检索节点（所有路由的必经之路），保证 faq_direct / rag_chat / chat 均能触发手势。
    nav_result = await _maybe_pre_run_navigation(user_input, available_tools, session_id)
    t_nav = time.monotonic()
    nav_gesture = ""
    nav_tool_results: list[dict] = []
    if nav_result:
        nav_gesture = nav_result.get("gesture", "")
        nav_tool_results = [{"tool_name": "hospital_navigate", "result": nav_result}]
        # gesture 已通过 state 传递，清除 _last_gesture 避免 reply_generator 重复弹出
        _last_gesture.pop(session_id, None)

    # 双集合检索：返回 (results, source_type)
    # 使用会话级缓存（针对追问场景）
    async def _do_search():
        return await dual_retriever.search(
            query=user_input,
            skill_ids=skill_ids,
            scene_id=scene_id,
            device_id=device_id,
            platform=platform,
        )

    results, source_type = await search_cache.get_or_search(session_id, user_input, _do_search)
    t_search = time.monotonic()

    if not results:
        logger.info("RAG 检索: 无结果")
        # 仍然尝试读取跨设备上下文
        cross_ctx_text = await _load_cross_device_context(scene_id, device_id)
        timings["rag"] = {
            "total": int((time.monotonic() - t_node_start) * 1000),
            "nav": int((t_nav - t_node_start) * 1000),
            "search": int((t_search - t_nav) * 1000),
            "rerank": 0,
        }
        ret = {"context": [], "max_rerank_score": 0.0, "route": "chat",
               "source_type": "knowledge", "cross_device_context": cross_ctx_text,
               "tool_results": nav_tool_results, "timings": timings}
        if nav_gesture:
            ret["gesture"] = nav_gesture
        return ret

    # Reranker 重排（可选）
    if settings.rag.rerank_enabled:
        results = await reranker.rerank(user_input, results, top_k=settings.rag.top_k)
        t_rerank = time.monotonic()
        max_score = max(r.get("rerank_score", 0.0) for r in results) if results else 0.0
        route = reranker.route(max_score, source_type=source_type)
        rerank_time = int((t_rerank - t_search) * 1000)
    else:
        # 不使用 Rerank，直接用向量得分路由
        max_score = max(r.get("score", 0.0) for r in results) if results else 0.0
        # 简化路由逻辑：>=0.7 → RAG+LLM，<0.7 → 普通对话
        if source_type == "faq" and max_score >= settings.rag.faq_direct_threshold:
            route = "faq_direct"
        elif max_score >= settings.rag.rerank_threshold_b:
            route = "rag_chat"
        else:
            route = "chat"
        rerank_time = 0
        logger.info("RAG 检索（无Rerank）: max_score=%.4f, route=%s", max_score, route)

    t_rerank = time.monotonic()

    # 读取跨设备上下文
    cross_ctx_text = await _load_cross_device_context(scene_id, device_id)

    timings["rag"] = {
        "total": int((time.monotonic() - t_node_start) * 1000),
        "nav": int((t_nav - t_node_start) * 1000),
        "search": int((t_search - t_nav) * 1000),
        "rerank": rerank_time,
    }

    logger.info(
        "RAG 检索+重排: results=%d, max_score=%.4f, route=%s, source_type=%s, cross_ctx=%s, nav_gesture=%s",
        len(results), max_score, route, source_type,
        "yes" if cross_ctx_text else "no", nav_gesture or "-",
    )

    ret = {"context": results, "max_rerank_score": max_score, "route": route,
           "source_type": source_type, "cross_device_context": cross_ctx_text,
           "tool_results": nav_tool_results, "timings": timings}
    if nav_gesture:
        ret["gesture"] = nav_gesture
    return ret


async def _load_cross_device_context(scene_id: str, device_id: str) -> str:
    """读取跨设备上下文"""
    if not scene_id or not device_id:
        return ""
    try:
        from app.session.cross_context import cross_context
        ctx = await cross_context.get(scene_id, device_id)
        if ctx:
            return cross_context.format_prompt(ctx)
    except Exception as e:
        logger.error("跨设备上下文读取异常: %s", e)
    return ""


def route_by_score(state: WorkflowState) -> str:
    """条件边：根据 RAG 重排分数和来源路由到不同节点

    - "noise" → 极低分，判定为噪音/无关输入，不送 LLM
    - "faq_direct" → FAQ 高相关，直接输出 LLM 润色后的回答
    - "rag_chat" → 知识库中/高相关，LLM 生成自然回答
    - "chat" → 低相关，普通对话
    """
    max_score = state.get("max_rerank_score", 0.0)

    # 无检索结果时直接走普通对话
    if max_score == 0.0:
        return "chat"

    # L2 噪音判定：rerank 最高分低于阈值 → 不送 LLM
    from app.config.settings import settings
    if max_score < settings.rag.rerank_noise_threshold:
        logger.info("L2 噪音判定: max_score=%.4f < %.4f → noise", max_score, settings.rag.rerank_noise_threshold)
        return "noise"

    source_type = state.get("source_type", "knowledge")
    return reranker.route(max_score, source_type=source_type)


async def direct_answer_node(state: WorkflowState) -> dict:
    """直接回答节点：FAQ 高相关命中时直接输出 LLM 润色后的回答，零 LLM 调用

    流程：
    1. 取 rerank_score 最高的片段文本（FAQ 中为润色后的自然语言回答）
    2. TTS 缓存优先（PostgreSQL）→ 未命中则合成并缓存
    3. 发送 ai.stream + tts.audio（缓存音频一次性发送）
    """
    start = time.monotonic()
    context = state.get("context", [])
    session_id = state.get("session_id", "")

    if not context:
        return {"reply": ""}

    # 检查中断
    if await _check_interrupted(session_id):
        logger.info("直接回答节点被中断: session_id=%s", session_id)
        return {"reply": "", "interrupted": True}

    # 取最高分片段
    best = max(context, key=lambda x: x.get("rerank_score", 0.0))
    reply = best.get("text", "").strip()

    if not reply:
        return {"reply": ""}

    logger.info("直接输出知识片段: session_id=%s, score=%.4f, text=%s",
                session_id, best.get("rerank_score", 0.0), reply[:30])

    # 发送 ai.stream（一次性完整输出）
    await _send({
        "type": "ai.stream",
        "payload": {"sessionId": session_id, "text": reply, "done": False},
    })
    await _send({
        "type": "ai.stream",
        "payload": {"sessionId": session_id, "text": "", "done": True},
    })

    # TTS：缓存优先
    t_tts = time.monotonic()
    try:
        audio_data = await tts_cache.get_or_synthesize(reply)
        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            # 缓存音频一次性发送
            await _send({
                "type": "tts.audio",
                "payload": {
                    "sessionId": session_id,
                    "audio": audio_b64,
                    "index": 0,
                    "format": "pcm",
                    "done": False,
                },
            })
    except Exception as e:
        logger.error("TTS 缓存获取/合成失败: session_id=%s, error=%s", session_id, e)
    tts_ms = int((time.monotonic() - t_tts) * 1000)

    # 发送 TTS 完成标记
    await _send({
        "type": "tts.audio",
        "payload": {
            "sessionId": session_id,
            "audio": "",
            "index": 1,
            "format": "pcm",
            "done": True,
        },
    })

    # 记录对话日记
    chat_logger.log(
        session_id=session_id,
        user_input=state.get("user_input", ""),
        reply=reply,
        route="faq_direct",
        rerank_score=best.get("rerank_score", 0.0),
        rag_hits=[c.get("text", "") for c in context],
        latency_ms=int((time.monotonic() - start) * 1000),
        skill_ids=state.get("skill_ids", []),
        asr_text=state.get("asr_text") or None,
    )

    # 生成跨设备上下文（FAQ 直接回答中提及其他楼层时）
    scene_id = state.get("scene_id", "")
    device_id = state.get("device_id", "")
    device_location = state.get("device_location", "")
    if scene_id and device_id and reply:
        try:
            from app.session.cross_context import cross_context
            await cross_context.generate(
                scene_id=scene_id,
                device_id=device_id,
                device_location=device_location,
                user_input=state.get("user_input", ""),
                reply=reply,
            )
        except Exception as e:
            logger.error("跨设备上下文生成异常(direct): %s", e)

    # 合并分阶段耗时：FAQ 直答无 LLM，仅记 TTS（缓存优先）
    timings = dict(state.get("timings", {}))
    timings["tts"] = {"total": tts_ms, "sentences": 1}

    return {"reply": reply, "timings": timings}


async def noise_response_node(state: WorkflowState) -> dict:
    """噪音回复节点（L2 Rerank 阈值判定为噪音）：不送 LLM，直接回固定文案"""
    from app.voice.noise_filter import NOISE_REPLY_TEXT

    start = time.monotonic()
    session_id = state.get("session_id", "")
    reply = NOISE_REPLY_TEXT

    logger.info("噪音回复: session_id=%s, max_score=%.4f", session_id, state.get("max_rerank_score", 0.0))

    # 发送 ai.stream
    await _send({
        "type": "ai.stream",
        "payload": {"sessionId": session_id, "text": reply, "done": False},
    })
    await _send({
        "type": "ai.stream",
        "payload": {"sessionId": session_id, "text": "", "done": True},
    })

    # TTS：缓存优先
    t_tts = time.monotonic()
    try:
        audio_data = await tts_cache.get_or_synthesize(reply)
        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            await _send({
                "type": "tts.audio",
                "payload": {
                    "sessionId": session_id,
                    "audio": audio_b64,
                    "index": 0,
                    "format": "pcm",
                    "done": False,
                },
            })
    except Exception as e:
        logger.error("噪音回复 TTS 失败: session_id=%s, error=%s", session_id, e)
    tts_ms = int((time.monotonic() - t_tts) * 1000)

    await _send({
        "type": "tts.audio",
        "payload": {"sessionId": session_id, "audio": "", "index": 1, "format": "pcm", "done": True},
    })

    # 记录对话日记
    chat_logger.log(
        session_id=session_id,
        user_input=state.get("user_input", ""),
        reply=reply,
        route="noise",
        rerank_score=state.get("max_rerank_score", 0.0),
        rag_hits=[],
        latency_ms=int((time.monotonic() - start) * 1000),
        skill_ids=state.get("skill_ids", []),
        asr_text=state.get("asr_text") or None,
    )

    timings = dict(state.get("timings", {}))
    timings["tts"] = {"total": tts_ms, "sentences": 1}

    return {"reply": reply, "timings": timings}


async def tool_executor_node(state: WorkflowState) -> dict:
    """工具执行节点：执行指定的工具"""
    user_input = state.get("user_input", "")
    available_tools = state.get("available_tools", [])
    tool_calls = state.get("tool_calls", [])

    if not tool_calls:
        tool_calls = _infer_tool_calls(user_input, available_tools)

    # 构建工具上下文
    from app.skill.context import SkillContext
    _ctx = SkillContext(
        session_id=state.get("session_id", ""),
        platform=state.get("platform", "fixed_terminal"),
        user_id=state.get("user_id", ""),
        scene_id=state.get("scene_id", ""),
        device_location=state.get("device_location", ""),
    )

    results = []
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        arguments = tc.get("arguments", {})

        tool = tool_registry.get_tool(tool_name)
        if tool is None or tool.handler is None:
            logger.error("工具不存在或无处理器: %s", tool_name)
            results.append({"tool_name": tool_name, "error": f"工具 {tool_name} 不存在"})
            continue

        try:
            result = await sandbox_executor.execute(
                handler=tool.handler,
                arguments=arguments,
                tool_name=tool_name,
                permissions=tool.permissions,
                context=_ctx,
            )
            results.append({"tool_name": tool_name, "result": result})
            logger.info("工具执行成功: %s", tool_name)
        except Exception as e:
            results.append({"tool_name": tool_name, "error": str(e)})
            logger.error("工具执行失败: %s, error=%s", tool_name, e)

    # 从工具结果中提取 gesture 提示（如导诊方向）
    gesture = ""
    for r in results:
        res = r.get("result")
        if isinstance(res, dict) and res.get("gesture"):
            gesture = res["gesture"]
            break

    ret: dict = {"tool_results": results}
    if gesture:
        ret["gesture"] = gesture
    return ret


def _infer_tool_calls(user_input: str, available_tools: list[dict]) -> list[dict]:
    """根据用户输入推断要调用的工具"""
    tool_calls = []
    tool_names = []
    for tool_def in available_tools:
        func = tool_def.get("function", {})
        tool_names.append(func.get("name", ""))

    if "时间" in user_input or "几点" in user_input:
        if "current_time" in tool_names or not tool_names:
            tool_calls.append({"name": "current_time", "arguments": {"format": "iso"}})

    if "动作" in user_input or "挥手" in user_input:
        if "avatar_action" in tool_names or not tool_names:
            tool_calls.append({"name": "avatar_action", "arguments": {"state": "greeting"}})

    if not tool_calls and tool_names:
        tool_calls.append({"name": tool_names[0], "arguments": {}})

    return tool_calls


async def reply_generator(state: WorkflowState) -> dict:
    """回复生成节点：调用 LLM 生成回复（token 级流式推送 + 分句 TTS）

    两种场景共用此节点：
    - rag_chat：RAG 上下文注入 LLM
    - chat：普通对话，无上下文注入
    """
    start = time.monotonic()
    messages = state.get("messages", [])
    context = state.get("context", [])
    tool_results = state.get("tool_results", [])
    available_tools = state.get("available_tools", [])
    skill_ids = state.get("skill_ids", [])
    session_id = state.get("session_id", "")
    user_input = state.get("user_input", "")
    max_rerank_score = state.get("max_rerank_score", 0.0)

    # ── 检查断点续讲 ──
    from app.session.breakpoint import breakpoint_manager
    breakpoint = breakpoint_manager.consume(session_id)
    resume_prompt = ""
    if breakpoint:
        remaining_text = breakpoint.get_remaining_text()
        if remaining_text:
            resume_prompt = f"\n\n[续讲提示] 用户刚才插话提问，现在需要从断点续讲。之前讲到这里：'...{remaining_text[:50]}...'，请自然地续讲后面的内容。"
            logger.info("启用断点续讲: session_id=%s, remaining=%d字符", session_id, len(remaining_text))

    # ── 个性化讲解风格 ──
    from app.session.style import get_style_prompt
    user_type = state.get("user_type", "general")
    style_prompt = get_style_prompt(user_type)

    # 构造系统提示
    device_location = state.get("device_location", "")
    scene_id = state.get("scene_id", "")

    if device_location:
        system_prompt = (
            f"你是{device_location}的数字人助手小诺。回答规则：\n"
            "1. 记住对话历史，自然地延续话题，可以引用之前聊过的内容\n"
            "2. 回答自然流畅，像真人对话一样，一般2-3句话，复杂问题可以多说几句\n"
            "3. 不使用emoji\n"
            "4. 简单问题简短回答，复杂问题给出有用信息\n"
            "5. 不要每次都说'还有什么可以帮您'之类的话\n"
            f"6. 你目前在{device_location}，优先提供本区域的信息\n"
            "7. 如果涉及其他区域，给出具体指引方向"
            f"{resume_prompt}{style_prompt}"
        )
    else:
        system_prompt = (
            "你是数字人助手小诺。回答规则：\n"
            "1. 记住对话历史，自然地延续话题，可以引用之前聊过的内容\n"
            "2. 回答自然流畅，像真人对话一样，一般2-3句话，复杂问题可以多说几句\n"
            "3. 不使用emoji\n"
            "4. 简单问题简短回答，复杂问题给出有用信息\n"
            "5. 不要每次都说'还有什么可以帮您'之类的话"
            f"{resume_prompt}{style_prompt}"
        )

    if skill_ids:
        from app.skill.loader import skill_loader
        for skill_id in skill_ids:
            skill = skill_loader.get_skill(skill_id)
            if skill and skill.system_prompt:
                system_prompt += f"\n\n{skill.system_prompt}"

    # 注入 RAG 上下文（只有 rag_chat 路由才注入，chat 路由忽略 context）
    route = state.get("route", "chat")
    if route == "rag_chat" and context:
        # 只取 top 2 片段注入
        top_context = sorted(context, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:2]
        context_text = "\n".join([f"- {c.get('text', '')}" for c in top_context])
        system_prompt += f"\n\n参考知识：\n{context_text}"

    # 注入工具执行结果（含 rag_retriever 预执行的 hospital_navigate 位置查询结果）
    if tool_results:
        result_text = "\n".join([
            f"- {r.get('tool_name', '')}: {json.dumps(r.get('result', r.get('error', '')), ensure_ascii=False)}"
            for r in tool_results
        ])
        system_prompt += f"\n\n工具执行结果：\n{result_text}\n请基于以上结果回答用户的问题。"

    # 注入跨设备上下文
    cross_device_context = state.get("cross_device_context", "")
    if cross_device_context:
        system_prompt += cross_device_context

    # 构造消息列表（可配置上下文窗口，超出部分摘要压缩）
    from app.config.settings import settings
    context_window = settings.rag.context_window  # 默认 20 条 = 10 轮

    if len(messages) > context_window:
        # 超出窗口的旧消息做摘要压缩
        old_messages = messages[:-context_window]
        recent_messages = messages[-context_window:]

        # 优先使用缓存摘要（避免阻塞回复路径 1-3s）
        from app.session.history import conversation_history
        cached_summary, cached_count = await conversation_history.get_summary(session_id)

        if cached_summary and cached_count >= len(old_messages):
            # 缓存有效（生成时的消息数 >= 当前旧消息数，说明已覆盖）
            summary = cached_summary
        else:
            # 缓存未命中或已过期：先用旧缓存（如果有），后台刷新
            summary = cached_summary or ""
            asyncio.create_task(
                _background_summarize(session_id, old_messages)
            )

        if summary:
            system_prompt += f"\n\n之前的对话摘要：\n{summary}"
    else:
        recent_messages = messages

    full_messages = [{"role": "system", "content": system_prompt}] + recent_messages

    # 合并内置 MCP 工具到可用工具列表。
    # rag_chat 路由不绑工具：知识片段与 hospital_navigate 导航结果都已注入 system_prompt，
    # 再绑工具只会增大请求体/首字延迟，并可能让模型重复调用已预执行的导航工具、
    # 触发第二轮 LLM（实测显著抬高首字延迟）。故 rag_chat 走纯文本流式。
    if route == "rag_chat":
        all_tools = []
    else:
        all_tools = get_all_available_tools(available_tools) if available_tools else get_all_available_tools()

    # 流式生成回复 + 分句 TTS（有工具时走 agentic loop）
    full_reply = ""
    sentence_buffer = ""
    tts_index = 0
    _token_count = 0  # 用于降频中断检查
    _has_audio = "audio" in state.get("capabilities", ["text", "audio", "avatar"])

    # 计时：LLM 首字延迟/总耗时 + TTS 累积
    t_llm_start = time.monotonic()
    llm_ttft_ms: int | None = None
    tts_total_ms = 0
    tts_sentences = 0
    tts_cache_hits = 0

    # 绑定 session_id + 上下文到 tool executor（用于捕获 gesture + 工具上下文）
    from functools import partial
    from app.skill.context import SkillContext
    _tool_context = SkillContext(
        session_id=session_id,
        platform=state.get("platform", "fixed_terminal"),
        user_id=state.get("user_id", ""),
        scene_id=state.get("scene_id", ""),
        device_location=state.get("device_location", ""),
    )
    _executor = partial(_mcp_tool_executor, session_id=session_id, context=_tool_context)

    # 难度路由：rag_chat 答案简短且基于注入知识，用快模型大幅压低首字延迟；
    # 普通/复杂对话保留默认（更强）模型。fast_model 为空则不路由。
    from app.config.settings import settings as _settings
    _fast_model = _settings.llm.fast_model
    _use_model = _fast_model if (route == "rag_chat" and _fast_model) else None
    if _use_model:
        logger.info("LLM 难度路由: route=%s → fast_model=%s", route, _use_model)

    stream_source = (
        llm_service.stream_chat_with_tools(full_messages, all_tools, tool_executor=_executor)
        if all_tools
        else llm_service.stream_chat(full_messages, model_override=_use_model)
    )

    async for event in stream_source:
        # 每 5 个 token 检查一次中断标志（降低 Redis 调用频率）
        _token_count += 1
        if _token_count % 5 == 0 and await _check_interrupted(session_id):
            logger.info("LLM 流式生成被中断: session_id=%s", session_id)
            break

        if event.type == "text":
            if llm_ttft_ms is None:
                llm_ttft_ms = int((time.monotonic() - t_llm_start) * 1000)
            full_reply += event.content if isinstance(event.content, str) else ""
            sentence_buffer += event.content
            await _send({
                "type": "ai.stream",
                "payload": {
                    "sessionId": session_id,
                    "text": event.content,
                    "done": False,
                },
            })

            # 检测句子结束，立即启动 TTS（仅音频能力客户端）
            if _has_audio and _is_sentence_end(sentence_buffer):
                sentence_text = sentence_buffer.strip()
                if sentence_text:
                    tts_index, _ms, _hit = await _stream_tts_sentence(sentence_text, session_id, tts_index)
                    tts_total_ms += _ms
                    tts_sentences += 1
                    tts_cache_hits += 1 if _hit else 0
                sentence_buffer = ""

        elif event.type == "tool_call":
            await _send({
                "type": "ai.tool.call",
                "payload": {
                    "sessionId": session_id,
                    "toolName": event.tool_name,
                    "toolCallId": event.tool_call_id,
                },
            })

    # 流结束后，发送剩余句子缓冲区（仅音频能力客户端）
    if _has_audio and sentence_buffer.strip():
        tts_index, _ms, _hit = await _stream_tts_sentence(sentence_buffer.strip(), session_id, tts_index)
        tts_total_ms += _ms
        tts_sentences += 1
        tts_cache_hits += 1 if _hit else 0

    # 发送完成标记
    await _send({
        "type": "ai.stream",
        "payload": {
            "sessionId": session_id,
            "text": "",
            "done": True,
        },
    })

    # 发送 TTS 完成标记（仅音频能力客户端）
    if _has_audio:
        await _send({
            "type": "tts.audio",
            "payload": {
                "sessionId": session_id,
                "audio": "",
                "index": tts_index,
                "format": "pcm",
                "done": True,
            },
        })

    logger.info("回复生成完成: session_id=%s, reply_len=%d", session_id, len(full_reply))

    # 记录对话日记
    route = state.get("route", "chat")
    chat_logger.log(
        session_id=session_id,
        user_input=state.get("user_input", ""),
        reply=full_reply,
        route=route,
        rerank_score=max_rerank_score,
        rag_hits=[c.get("text", "") for c in context] if context else [],
        latency_ms=int((time.monotonic() - start) * 1000),
        skill_ids=skill_ids,
        asr_text=state.get("asr_text") or None,
    )

    # 生成跨设备上下文（回复中提及其他楼层时）
    scene_id = state.get("scene_id", "")
    device_id = state.get("device_id", "")
    device_location = state.get("device_location", "")
    if scene_id and device_id and full_reply:
        try:
            from app.session.cross_context import cross_context
            await cross_context.generate(
                scene_id=scene_id,
                device_id=device_id,
                device_location=device_location,
                user_input=state.get("user_input", ""),
                reply=full_reply,
            )
        except Exception as e:
            logger.error("跨设备上下文生成异常: %s", e)

    # 提取工具执行中捕获的 gesture
    gesture = _last_gesture.pop(session_id, "")

    # 合并分阶段耗时：LLM（首字/总）+ TTS（总/句数/缓存命中）
    timings = dict(state.get("timings", {}))
    llm_total_ms = int((time.monotonic() - t_llm_start) * 1000)
    timings["llm"] = {"ttft": llm_ttft_ms or 0, "total": llm_total_ms}
    timings["tts"] = {"total": tts_total_ms, "sentences": tts_sentences, "cache_hits": tts_cache_hits}

    ret: dict = {"reply": full_reply, "timings": timings}
    if gesture:
        ret["gesture"] = gesture
    return ret


async def _stream_tts_sentence(text: str, session_id: str, start_index: int) -> tuple[int, int, bool]:
    """流式 TTS 合成一句话并发送（缓存优先）

    Args:
        text: 要合成的文本
        session_id: 会话 ID
        start_index: TTS 分片起始序号

    Returns:
        (下一个可用的 index, 本句耗时 ms, 是否缓存命中)
    """
    chunk_index = start_index
    t_start = time.monotonic()

    # 1. 查缓存（命中则一次性发送，延迟极低）
    try:
        cached_audio = await tts_cache.get(text)
        if cached_audio:
            audio_b64 = base64.b64encode(cached_audio).decode("ascii")
            await _send({
                "type": "tts.audio",
                "payload": {
                    "sessionId": session_id,
                    "audio": audio_b64,
                    "index": chunk_index,
                    "format": "pcm",
                    "done": False,
                },
            })
            logger.debug("TTS 缓存命中: session=%s, text=%s", session_id, text[:20])
            return chunk_index + 1, int((time.monotonic() - t_start) * 1000), True
    except Exception as e:
        logger.debug("TTS 缓存查询失败(降级为合成): %s", e)

    # 2. 未命中：流式合成 + 收集完整音频用于缓存
    audio_chunks: list[bytes] = []
    try:
        async for audio_chunk in tts_service.synthesize_stream(text):
            if audio_chunk:
                audio_chunks.append(audio_chunk)
                audio_b64 = base64.b64encode(audio_chunk).decode("ascii")
                await _send({
                    "type": "tts.audio",
                    "payload": {
                        "sessionId": session_id,
                        "audio": audio_b64,
                        "index": chunk_index,
                        "format": "pcm",
                        "done": False,
                    },
                })
                chunk_index += 1
    except Exception as e:
        logger.error("分句 TTS 合成失败: session_id=%s, text=%s, error=%s", session_id, text[:20], e)

    # 3. 异步写入缓存（不阻塞主流程）
    if audio_chunks:
        import asyncio
        full_audio = b"".join(audio_chunks)
        duration_ms = int(len(full_audio) / 32000 * 1000)

        async def _cache_put():
            try:
                await tts_cache.put(text, full_audio, duration_ms=duration_ms)
            except Exception as e:
                logger.debug("TTS 缓存写入失败: %s", e)

        asyncio.create_task(_cache_put())

    return chunk_index, int((time.monotonic() - t_start) * 1000), False


async def avatar_driver_node(state: WorkflowState) -> dict:
    """Avatar 驱动节点：根据对话状态生成视频切换指令

    状态优先级：
      1. state 中有 gesture 字段（来自 Skill 工具调用）→ 使用指定手势
      2. 默认使用 talking 状态（回复/讲解中）

    若客户端未声明 avatar 能力则跳过。
    """
    capabilities = state.get("capabilities", ["text", "audio", "avatar"])
    if "avatar" not in capabilities:
        return {"drive_data": {}}

    reply = state.get("reply", "")
    session_id = state.get("session_id", "")
    gesture = state.get("gesture", "")  # Skill 可设置: point_left/point_right/bow/自定义
    avatar_config = state.get("avatar_config", None)  # 形象配置（含 gestures 列表）

    # 确定视频状态：有 gesture 就用，否则 talking
    avatar_state = gesture if gesture else "talking"

    # 使用 per-session driver（从形象配置构造，未知 gesture 自动降级 talking）
    driver = get_session_driver(session_id, avatar_config)
    drive_data = await driver.generate_drive(avatar_state, reply)

    await _send({
        "type": "avatar.drive",
        "payload": {
            "sessionId": session_id,
            **drive_data,
        },
    })

    return {"drive_data": drive_data}


async def faq_candidate_recorder_node(state: WorkflowState) -> dict:
    """FAQ 自动晋升已关闭：所有对话走人工审核，不再自动记录候选。"""
    return {"faq_candidate_recorded": False}

