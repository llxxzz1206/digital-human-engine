from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator

from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """LLM 流式输出事件"""

    type: str  # "text" | "tool_call" | "done"
    content: str | dict = ""
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class ToolCallInfo:
    """LLM 返回的工具调用信息（用于聚合流式片段）"""

    id: str = ""
    name: str = ""
    arguments: str = ""  # JSON string


class LLMService:
    def __init__(self) -> None:
        self._provider = settings.llm.provider
        self._model = settings.llm.model
        self._api_key = settings.llm.api_key
        self._api_base = settings.llm.api_base
        self._client = None

        # 智能路由：简单问题用 fast_model
        self._fast_model = settings.llm.fast_model if hasattr(settings.llm, "fast_model") else ""
        self._use_fast_model = bool(self._fast_model)

        # Fallback providers
        self._fallback_clients: list[dict] = []  # [{"client": AsyncOpenAI, "model": str, "name": str}]
        self._failure_counts: dict[str, int] = {}  # provider_name -> consecutive failures
        self._circuit_until: dict[str, float] = {}  # provider_name -> timestamp to skip until
        self._parse_fallback_providers()

        if self._provider != "mock" and self._api_key:
            self._init_openai_client()

    def _parse_fallback_providers(self) -> None:
        """解析备用 LLM 配置"""
        raw = settings.llm.fallback_providers.strip()
        if not raw:
            return
        try:
            providers = json.loads(raw)
            for p in providers:
                self._fallback_clients.append({
                    "model": p.get("model", ""),
                    "api_key": p.get("api_key", ""),
                    "api_base": p.get("api_base", ""),
                    "name": p.get("name", p.get("model", "fallback")),
                    "client": None,  # lazy init
                })
            logger.info("已配置 %d 个备用 LLM provider", len(self._fallback_clients))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("fallback_providers 解析失败: %s", e)

    def _get_fallback_client(self, fb: dict):
        """懒初始化 fallback client"""
        if fb["client"] is None and fb["api_key"]:
            from openai import AsyncOpenAI
            kwargs: dict = {"api_key": fb["api_key"]}
            if fb["api_base"]:
                kwargs["base_url"] = fb["api_base"]
            fb["client"] = AsyncOpenAI(**kwargs)
        return fb["client"]

    def _is_circuit_open(self, name: str) -> bool:
        """熔断检查：连续 5 次失败后 60s 内跳过"""
        import time
        until = self._circuit_until.get(name, 0)
        if time.time() < until:
            return True
        # 熔断恢复
        if name in self._circuit_until:
            del self._circuit_until[name]
            self._failure_counts[name] = 0
        return False

    def _record_failure(self, name: str) -> None:
        import time
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        if self._failure_counts[name] >= 5:
            self._circuit_until[name] = time.time() + 60
            logger.warning("LLM provider '%s' 熔断 60s（连续 %d 次失败）", name, self._failure_counts[name])

    def _record_success(self, name: str) -> None:
        self._failure_counts[name] = 0

    def _init_openai_client(self) -> None:
        from openai import AsyncOpenAI

        kwargs: dict = {"api_key": self._api_key}
        if self._api_base:
            kwargs["base_url"] = self._api_base
        self._client = AsyncOpenAI(**kwargs)
        logger.info("OpenAI 客户端已初始化, model=%s, base_url=%s", self._model, self._api_base)
        if self._fast_model:
            logger.info("Fast model 已配置: %s", self._fast_model)

    async def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model_override: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式对话，yield StreamEvent（文本/工具调用/完成）

        model_override: 指定本次调用使用的模型（难度路由），None 用默认 model。
        """
        if self._provider == "mock":
            async for event in self._mock_stream(tools):
                yield event
        else:
            async for event in self._openai_stream(messages, tools, model_override):
                yield event

    async def stream_chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_executor=None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """带工具执行的 Agentic Loop：LLM→tool_call→execute→回传LLM→重复直到无tool_calls

        Args:
            messages: 对话历史
            tools: 可用工具定义（OpenAI function calling 格式）
            tool_executor: 异步函数 (tool_name, args) -> result
        """
        max_iterations = 5
        current_messages = list(messages)

        for _iteration in range(max_iterations):
            has_tool_calls = False
            tool_calls_collected: dict[int, ToolCallInfo] = {}
            full_content = ""

            async for event in self.stream_chat(current_messages, tools):
                if event.type == "text":
                    full_content += event.content if isinstance(event.content, str) else ""
                    yield event
                elif event.type == "tool_call":
                    has_tool_calls = True
                    # 按 tool_call_id 聚合片段
                    tc_id = event.tool_call_id or ""
                    if tc_id not in tool_calls_collected:
                        tool_calls_collected[tc_id] = ToolCallInfo(
                            id=tc_id,
                            name=event.tool_name or "",
                        )
                    if isinstance(event.content, str):
                        tool_calls_collected[tc_id].arguments += event.content
                    yield event

            if not has_tool_calls:
                yield StreamEvent(type="done")
                return

            # 构造 assistant 消息（含 tool_calls）
            assistant_msg: dict = {"role": "assistant", "content": full_content or None}
            openai_tool_calls = []
            for tc in tool_calls_collected.values():
                openai_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                })
            assistant_msg["tool_calls"] = openai_tool_calls
            current_messages.append(assistant_msg)

            # 执行每个工具并追加结果到 messages
            for tc in tool_calls_collected.values():
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                if tool_executor:
                    try:
                        result = await tool_executor(tc.name, args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": "无工具执行器"}

                result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        # 达到最大迭代次数
        yield StreamEvent(type="done")

    async def _mock_stream(
        self,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Mock 模式流式输出"""
        if tools:
            first_func = tools[0].get("function", {}) if tools else {}
            tool_name = first_func.get("name", "echo")
            yield StreamEvent(
                type="tool_call",
                content='{"text": "hello"}',
                tool_call_id="call_mock_001",
                tool_name=tool_name,
            )

        mock_reply = "你好！我是数字人助手，很高兴为您服务。请问有什么可以帮助您的？"
        for char in mock_reply:
            yield StreamEvent(type="text", content=char)
            await asyncio.sleep(0.05)
        yield StreamEvent(type="done")

    async def _openai_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model_override: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """OpenAI 兼容 API 流式输出，支持 fallback + 熔断"""
        if self._client is None:
            logger.error("OpenAI 客户端未初始化，回退到 Mock")
            async for event in self._mock_stream(tools):
                yield event
            return

        model = model_override or self._model

        # 构建尝试列表：主 provider + fallbacks
        attempts_list: list[tuple[str, object, str]] = []  # (name, client, model)
        primary_name = f"primary:{self._provider}"
        if not self._is_circuit_open(primary_name):
            attempts_list.append((primary_name, self._client, model))
        for fb in self._fallback_clients:
            if not self._is_circuit_open(fb["name"]):
                client = self._get_fallback_client(fb)
                if client:
                    attempts_list.append((fb["name"], client, fb["model"]))

        if not attempts_list:
            logger.error("所有 LLM provider 均处于熔断状态")
            yield StreamEvent(type="text", content="抱歉，我暂时无法回答，请稍后再试。")
            yield StreamEvent(type="done")
            return

        for provider_name, client, use_model in attempts_list:
            try:
                kwargs: dict = {
                    "model": use_model,
                    "messages": messages,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools

                stream = await client.chat.completions.create(**kwargs)

                # 聚合 tool_calls 流式片段
                current_tool_calls: dict[int, ToolCallInfo] = {}

                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # 文本内容
                    if delta.content:
                        yield StreamEvent(type="text", content=delta.content)

                    # tool_calls
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in current_tool_calls:
                                current_tool_calls[idx] = ToolCallInfo(
                                    id=tc_delta.id or "",
                                    name=tc_delta.function.name if tc_delta.function else "",
                                )
                            else:
                                if tc_delta.id:
                                    current_tool_calls[idx].id = tc_delta.id
                                if tc_delta.function and tc_delta.function.name:
                                    current_tool_calls[idx].name = tc_delta.function.name

                            # 追加 arguments 片段并推送
                            if tc_delta.function and tc_delta.function.arguments:
                                current_tool_calls[idx].arguments += tc_delta.function.arguments
                                yield StreamEvent(
                                    type="tool_call",
                                    content=tc_delta.function.arguments,
                                    tool_call_id=current_tool_calls[idx].id,
                                    tool_name=current_tool_calls[idx].name,
                                )

                # 成功
                self._record_success(provider_name)
                yield StreamEvent(type="done")
                return

            except Exception as e:
                self._record_failure(provider_name)
                logger.warning("LLM provider '%s' 请求失败: %s", provider_name, e)
                continue  # 尝试下一个 provider

        # 所有 provider 都失败
        logger.error("所有 LLM provider 均失败")
        yield StreamEvent(type="text", content="抱歉，我暂时无法回答，请稍后再试。")
        yield StreamEvent(type="done")

    async def chat(self, messages: list[dict]) -> str:
        """非流式对话"""
        if self._provider == "mock":
            return "你好！我是数字人助手，很高兴为您服务。请问有什么可以帮助您的？"

        if self._client is None:
            return "你好！我是数字人助手，很高兴为您服务。请问有什么可以帮助您的？"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM 非流式请求失败: %s", e)
            return ""


llm_service = LLMService()
