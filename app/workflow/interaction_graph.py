from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

from langgraph.graph import END, StateGraph

from app.config.settings import settings
from app.infrastructure.redis import RedisPool
from app.services.latency_alerter import latency_alerter
from app.session.manager import session_manager
from app.workflow.nodes import (
    avatar_driver_node,
    direct_answer_node,
    faq_candidate_recorder_node,
    noise_response_node,
    rag_retriever_node,
    reply_generator,
    route_by_score,
    set_send_func,
)
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def _format_timing(session_id: str, route: str, timings: dict, wf_ms: int) -> str:
    """把各节点写入 state["timings"] 的耗时拼成一行 [TIMING] 日志。

    timings 结构（各节点按需写入，缺失即跳过）：
      asr : int(ms)
      rag : {"total","nav","search","rerank"}
      llm : {"ttft","total"}
      tts : {"total","sentences","cache_hits"}
    """
    parts: list[str] = []

    asr = timings.get("asr")
    if asr is not None:
        parts.append(f"ASR={asr}ms")

    rag = timings.get("rag")
    if rag:
        parts.append(
            f"RAG={rag.get('total', 0)}ms "
            f"(nav={rag.get('nav', 0)} search={rag.get('search', 0)} rerank={rag.get('rerank', 0)})"
        )

    llm = timings.get("llm")
    if llm:
        ttft = llm.get("ttft")
        if ttft is not None:
            parts.append(f"LLM={llm.get('total', 0)}ms (ttft={ttft})")
        else:
            parts.append(f"LLM={llm.get('total', 0)}ms")

    tts = timings.get("tts")
    if tts:
        cache_hits = tts.get("cache_hits")
        cache_str = f"/缓存{cache_hits}" if cache_hits is not None else ""
        parts.append(f"TTS={tts.get('total', 0)}ms ({tts.get('sentences', 0)}句{cache_str})")

    # 端到端 = ASR(若有) + 工作流
    e2e = wf_ms + (asr or 0)
    detail = " | ".join(parts) if parts else "无分阶段数据"
    return (f"[TIMING] session={session_id} route={route} | {detail} | "
            f"工作流={wf_ms}ms 端到端={e2e}ms")


# 性能明细在 Redis 中的存放：list，保留最近 TIMING_KEEP 条，供管理后台查询
TIMING_KEY = "digitalhuman:timings"
TIMING_KEEP = 200


async def _record_timing(session_id: str, route: str, timings: dict, wf_ms: int) -> None:
    """把本次对话的分阶段耗时 flatten 成单条记录写入 Redis（供管理后台性能监控）。

    只记数、不阻塞主流程：任何异常都吞掉，绝不影响对话。
    """
    try:
        rag = timings.get("rag") or {}
        llm = timings.get("llm") or {}
        tts = timings.get("tts") or {}
        asr = timings.get("asr")

        # 各阶段"出处"：记录当时用的模型/引擎，便于切换模型后对照耗时
        asr_cfg = settings.asr
        if asr_cfg.provider == "whisper":
            asr_tag = f"whisper·{asr_cfg.model}·{asr_cfg.device}"
        else:
            asr_tag = asr_cfg.provider  # xunfei / disabled

        record = {
            "ts": int(time.time() * 1000),
            "sessionId": session_id,
            "route": route,
            "asr": asr,
            "ragTotal": rag.get("total"),
            "ragNav": rag.get("nav"),
            "ragSearch": rag.get("search"),
            "ragRerank": rag.get("rerank"),
            "llmTotal": llm.get("total"),
            "llmTtft": llm.get("ttft"),
            "ttsTotal": tts.get("total"),
            "ttsSentences": tts.get("sentences"),
            "ttsCacheHits": tts.get("cache_hits"),
            "workflow": wf_ms,
            "e2e": wf_ms + (asr or 0),
            # 出处（模型/引擎版本）
            "llmModel": settings.llm.model,
            "asrTag": asr_tag,
            "ttsVoice": settings.tts.voice,
        }
        r = await RedisPool.get()
        pipe = r.pipeline()
        pipe.lpush(TIMING_KEY, json.dumps(record, ensure_ascii=False))
        pipe.ltrim(TIMING_KEY, 0, TIMING_KEEP - 1)
        await pipe.execute()
    except Exception as e:  # noqa: BLE001 - 记录耗时失败不应影响对话
        logger.warning("写入性能明细失败(忽略): %s", e)


class InteractionGraph:
    """基于 LangGraph 的交互工作流

    状态图（双集合架构）：
    START → rag_retriever → route_by_score
                              ├─ "faq_direct" → direct_answer → avatar_driver → END
                              ├─ "rag_chat"   → reply_generator → faq_recorder → avatar_driver → END
                              └─ "chat"       → reply_generator → avatar_driver → END

    路由规则（来源感知 + 双阈值）：
    - FAQ 高分命中 (≥0.85): faq_direct，零 LLM 调用
    - 知识库中/高分 (≥0.4): rag_chat，LLM 生成自然回答
    - 低相关 (<0.4): chat，普通对话
    """

    def __init__(self) -> None:
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()
        # session 级互斥锁：同一 session 同时只允许一个 workflow 运行
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # 保护 _session_locks 字典
        # 存储 workflow 最后一次回复（用于写入对话历史）
        self._last_replies: dict[str, str] = {}

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """获取 session 级锁"""
        async with self._locks_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        graph = StateGraph(WorkflowState)

        # 添加节点
        graph.add_node("rag_retriever", rag_retriever_node)
        graph.add_node("direct_answer", direct_answer_node)
        graph.add_node("noise_response", noise_response_node)
        graph.add_node("reply_generator", reply_generator)
        graph.add_node("faq_recorder", faq_candidate_recorder_node)
        graph.add_node("avatar_driver", avatar_driver_node)

        # 设置入口
        graph.set_entry_point("rag_retriever")

        # 条件边：RAG 重排分数 + 来源 → 不同节点
        graph.add_conditional_edges(
            "rag_retriever",
            route_by_score,
            {
                "faq_direct": "direct_answer",
                "noise": "noise_response",
                "rag_chat": "reply_generator",
                "chat": "reply_generator",
            },
        )

        # FAQ direct / 噪音回复 → Avatar 驱动
        graph.add_edge("direct_answer", "avatar_driver")
        graph.add_edge("noise_response", "avatar_driver")

        # 回复生成 → FAQ 候选记录 → Avatar 驱动
        graph.add_edge("reply_generator", "faq_recorder")
        graph.add_edge("faq_recorder", "avatar_driver")

        # Avatar 驱动 → 结束
        graph.add_edge("avatar_driver", END)

        return graph

    # 锁等待超时（秒）：workflow 正常不超过 60s，超时说明前一个 workflow 卡死
    _LOCK_TIMEOUT = 90

    async def run(
        self,
        session_id: str,
        messages: list[dict],
        skill_ids: list[str] | None = None,
        tools: list[dict] | None = None,
        send_func: Callable[[dict], Awaitable[None]] | None = None,
        asr_text: str | None = None,
        scene_id: str = "",
        device_id: str = "",
        device_location: str = "",
        user_type: str = "general",
        asr_ms: int | None = None,
    ) -> None:
        """运行交互工作流（带 session 级互斥锁）

        同一 session 同时只允许一个 workflow 运行。
        新请求到来时，如果旧 workflow 还在运行，会等待锁释放后再执行。
        锁等待超过 _LOCK_TIMEOUT 秒则放弃（防止前一个 workflow 卡死导致永久阻塞）。
        """
        lock = await self._get_lock(session_id)

        if lock.locked():
            logger.warning("Session %s 已有正在运行的 workflow，新请求排队等待", session_id)

        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._LOCK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Session %s 锁等待超时(%ds)，前一个 workflow 可能卡死，跳过本次执行",
                         session_id, self._LOCK_TIMEOUT)
            # 强制释放锁（前一个 workflow 已无意义）
            try:
                lock.release()
            except RuntimeError:
                pass
            return

        try:
            await self._run_workflow(session_id, messages, skill_ids, tools, send_func, asr_text,
                                     scene_id, device_id, device_location, user_type, asr_ms)
        finally:
            lock.release()

    async def _run_workflow(
        self,
        session_id: str,
        messages: list[dict],
        skill_ids: list[str] | None,
        tools: list[dict] | None,
        send_func: Callable[[dict], Awaitable[None]] | None,
        asr_text: str | None,
        scene_id: str = "",
        device_id: str = "",
        device_location: str = "",
        user_type: str = "general",
        asr_ms: int | None = None,
    ) -> None:
        """实际执行工作流"""
        # 设置模块级 send_func
        set_send_func(send_func)

        skill_ids = skill_ids or []
        user_input = ""
        if messages:
            last_msg = messages[-1]
            user_input = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

        # 初始 timings：语音输入时带上 ASR 耗时
        timings: dict = {}
        if asr_ms is not None:
            timings["asr"] = asr_ms

        # 获取会话信息
        session = await session_manager.get_session(session_id)

        # 读取形象配置（gestures/loopStates），供 avatar_driver_node 使用
        avatar_config = None
        avatar_id = getattr(session, "avatarId", "") or getattr(session, "avatar_id", "")
        if avatar_id:
            try:
                r = await RedisPool.get()
                raw = await r.hgetall(f"digitalhuman:avatar:{avatar_id}")
                if raw:
                    avatar_config = raw
            except Exception:
                pass  # Redis 不可用时用默认动作集

        # 初始状态
        initial_state: WorkflowState = {
            "session_id": session_id,
            "user_input": user_input,
            "messages": messages,
            "skill_ids": skill_ids,
            "platform": getattr(session, "platform", "fixed_terminal"),
            "capabilities": getattr(session, "capabilities", ["text", "audio", "avatar"]),
            "scene_id": scene_id,
            "device_id": device_id,
            "device_location": device_location,
            "currentLocation": getattr(session, "currentLocation", ""),
            "locationSource": getattr(session, "locationSource", ""),
            "user_type": user_type,
            "context": [],
            "max_rerank_score": 0.0,
            "route": "chat",
            "source_type": "knowledge",
            "tool_calls": [],
            "tool_results": [],
            "available_tools": tools or [],
            "reply": "",
            "asr_text": asr_text or "",
            "drive_data": {},
            "interrupted": False,
            "cross_device_context": "",
            "timings": timings,
            "avatar_config": avatar_config,
        }

        logger.info("InteractionGraph 运行: session_id=%s, skill_ids=%s, scene=%s, device=%s, input='%s'",
                     session_id, skill_ids, scene_id, device_id, user_input[:30])

        t_wf_start = time.monotonic()
        try:
            # 执行工作流
            result = await self._compiled.ainvoke(initial_state)
            reply = result.get("reply", "")
            logger.info("InteractionGraph 完成: session_id=%s, max_score=%.4f, reply_len=%d",
                        session_id, result.get("max_rerank_score", 0), len(reply))
            # 分阶段耗时汇总（定位 ASR/RAG/LLM/TTS 瓶颈）
            wf_ms = int((time.monotonic() - t_wf_start) * 1000)
            route = result.get("route", "chat")
            timings = result.get("timings", {})
            logger.info(_format_timing(session_id, route, timings, wf_ms))
            # 同步写入 Redis，供管理后台性能监控查询
            await _record_timing(session_id, route, timings, wf_ms)
            
            # 延迟告警检查
            alert_result = await latency_alerter.check(session_id, timings, route)
            if alert_result.get("alerts"):
                logger.warning("延迟告警: session=%s, alerts=%d", session_id, alert_result["alert_count"])
            # 存储 reply 供外部读取（用于写入对话历史）
            self._last_replies[session_id] = reply
        except Exception as e:
            logger.error("InteractionGraph 异常: session_id=%s, error=%s", session_id, e)
            # 异常时也发送完成标记，避免客户端挂起
            if send_func:
                await send_func({
                    "type": "ai.stream",
                    "payload": {
                        "sessionId": session_id,
                        "text": "",
                        "done": True,
                    },
                })
        finally:
            # 清理 send_func
            set_send_func(None)

    async def interrupt(self, session_id: str) -> None:
        """中断正在运行的工作流"""
        try:
            redis = await RedisPool.get()
            key = f"digitalhuman:interrupt:{session_id}"
            await redis.set(key, "1", ex=60)
            logger.info("工作流中断标志已设置: session_id=%s", session_id)
        except Exception as e:
            logger.error("设置中断标志失败: %s", e)

    def get_last_reply(self, session_id: str) -> str:
        """获取 session 最后一次 workflow 的回复文本（非破坏性读取）"""
        return self._last_replies.get(session_id, "")

    def clear_last_reply(self, session_id: str) -> None:
        """清除 session 的最后一次回复（由历史记录完成后调用）"""
        self._last_replies.pop(session_id, None)

    def cleanup_session(self, session_id: str) -> None:
        """清理 session 相关资源（会话销毁时调用）"""
        self._last_replies.pop(session_id, None)
        self._session_locks.pop(session_id, None)


interaction_graph = InteractionGraph()
