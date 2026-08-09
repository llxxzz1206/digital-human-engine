from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from app.config.settings import settings
from app.mcp.registry import tool_registry
from app.mcp.sandbox import sandbox_executor
from app.rag.search_cache import search_cache
from app.services.user_manager import user_manager
from app.session.breakpoint import breakpoint_manager
from app.session.history import conversation_history
from app.session.manager import session_manager
from app.skill.loader import skill_loader
from app.voice.asr_service import asr_service
from app.voice.audio_enhancer import EnhancementConfig, get_audio_enhancer
from app.voice.kws_service import kws_service
from app.voice.noise_filter import NOISE_REPLY_TEXT, compute_rms, is_hallucination
from app.voice.tts_service import tts_service
from app.voice.wake_match import match_wake
from app.voice.xunfei_asr import XunfeiAsrSession, xunfei_asr_manager
from app.workflow.interaction_graph import interaction_graph
from app.workflow.nodes import clear_session_driver, get_session_driver

logger = logging.getLogger(__name__)

# 会话级音频分片缓冲区：session_id -> list[bytes]
_audio_buffers: dict[str, list[bytes]] = {}
# 讯飞流式 ASR 的 RMS 门控缓冲：session_id -> bytearray（仅用于说完后判定是否静音/底噪）
_asr_rms_buffers: dict[str, bytearray] = {}


async def handle_message(msg: dict, websocket: WebSocket) -> None:
    msg_type = msg.get("type", "")
    payload = msg.get("payload", {})
    request_id = msg.get("requestId")

    logger.info("处理消息: type=%s, requestId=%s", msg_type, request_id)

    match msg_type:
        case "session.create":
            await _handle_session_create(payload, websocket)
        case "session.destroy":
            await _handle_session_destroy(payload, websocket)
        case "skill.mount":
            await _handle_skill_mount(payload, websocket)
        case "skill.unmount":
            await _handle_skill_unmount(payload, websocket)
        case "chat.send":
            await _handle_chat_send(payload, websocket)
        case "interrupt":
            await _handle_interrupt(payload, websocket)
        case "greeting.trigger":
            await _handle_greeting_trigger(payload, websocket)
        case "ai.generate":
            await _handle_ai_generate(payload, websocket)
        case "audio.stream":
            await _handle_audio_stream(payload, websocket)
        case "audio.end":
            await _handle_audio_end(payload, websocket)
        case "tts.request":
            await _handle_tts_request(payload, websocket)
        case "asr.request":
            await _handle_asr_request(payload, websocket)
        case "mcp.tool.call":
            await _handle_mcp_tool_call(payload, request_id, websocket)
        case _:
            logger.warning("未知消息类型: %s", msg_type)


async def _send(websocket: WebSocket, message: dict) -> None:
    """通过 WS 连接发送消息（断连时静默丢弃）"""
    try:
        await websocket.send_json(message)
    except (WebSocketDisconnect, ConnectionError, OSError, RuntimeError):
        # WS 已断连或状态异常，静默丢弃消息（避免后台 task 崩溃）
        pass


async def _handle_session_create(payload: dict, websocket: WebSocket) -> None:
    """处理会话创建请求（支持固定终端和移动端）"""
    session_id = payload.get("sessionId") or str(uuid.uuid4())
    user_id = payload.get("userId", "")
    avatar_id = payload.get("avatarId", "")
    channel_id = payload.get("channelId", "")

    # 平台类型（默认固定终端）
    platform = payload.get("platform", "fixed_terminal")

    # 客户端能力声明（默认全能力）
    capabilities = payload.get("capabilities", ["text", "audio", "avatar"])

    # 固定终端参数
    scene_id = payload.get("sceneId", "")
    device_id = payload.get("deviceId", "")
    device_location = payload.get("deviceLocation", "")

    # 移动端参数
    current_location = payload.get("currentLocation", "")
    location_source = payload.get("locationSource", "")
    latitude = payload.get("latitude")  # 纬度
    longitude = payload.get("longitude")  # 经度

    # 用户类型（影响讲解风格）
    user_type = payload.get("userType", "general")

    # ── 用户认证（移动端必填）──
    if platform in ("mobile_app", "mini_app"):
        if not user_id:
            await websocket.send_json({
                "type": "error",
                "payload": {"code": "USER_ID_REQUIRED", "message": "移动端必须提供 userId"}
            })
            return
        if not await user_manager.validate_user(user_id):
            await websocket.send_json({
                "type": "error",
                "payload": {"code": "INVALID_USER_ID", "message": "无效的用户ID"}
            })
            return

    # ── 场景探测（移动端自动探测）──
    detected_scene = None
    if platform in ("mobile_app", "mini_app") and latitude and longitude:
        from app.services.scene_detector import GeoLocation, scene_detector

        geo_loc = GeoLocation(latitude=float(latitude), longitude=float(longitude))
        detected_scene = await scene_detector.detect(geo_loc, user_id)

        # 如果探测到场景，优先使用
        if detected_scene and not scene_id:
            scene_id = detected_scene
            logger.info("移动端场景自动探测: user=%s, scene=%s", user_id, scene_id)

    # ── 移动端：用户级会话管理（多设备同步）──
    if platform in ("mobile_app", "mini_app") and user_id:
        # 虚拟设备 ID（移动端）
        if not device_id:
            device_id = f"mobile_{user_id}"

        # 获取或创建用户会话
        session, is_new = await user_manager.get_or_create_session(
            user_id=user_id,
            device_id=device_id,
            platform=platform,
            scene_id=scene_id,
        )
        session_id = session.sessionId

        # 如果探测到新场景，更新会话
        if detected_scene and session.sceneId != detected_scene:
            session.sceneId = detected_scene
            session.updatedAt = int(time.time() * 1000)
            await session_manager.save_session(session)

        # 更新用户偏好
        if detected_scene:
            await user_manager.update_user_preference(user_id, last_active_scene=detected_scene)
    else:
        # ── 固定终端：传统会话创建 ──
        session = await session_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            avatar_id=avatar_id,
            scene_id=scene_id,
            device_id=device_id,
            device_location=device_location,
            user_type=user_type,
            platform=platform,
            current_location=current_location,
            location_source=location_source,
            capabilities=capabilities,
        )

    # 自动挂载 sceneId 对应的 Skill（如果有）
    if scene_id:
        skill = skill_loader.get_skill(scene_id)
        if skill:
            await session_manager.mount_skill(session_id, scene_id)
            logger.info("自动挂载场景 Skill: session=%s, skill=%s", session_id, scene_id)
            # 重新获取 session 以包含挂载后的 skills
            session = await session_manager.get_session(session_id) or session

    await _send(websocket, {
        "type": "session.created",
        "payload": {
            "sessionId": session.sessionId,
            "userId": session.userId,
            "avatarId": session.avatarId,
            "mountedSkills": session.mountedSkills,  # 返回挂载后的 skills
            "sceneId": session.sceneId,
            "deviceId": session.deviceId,
            "userType": session.userType,
            "deviceLocation": session.deviceLocation,
            "channelId": channel_id,
        },
    })


async def _handle_session_destroy(payload: dict, websocket: WebSocket) -> None:
    """处理会话销毁请求"""
    session_id = payload.get("sessionId", "")
    channel_id = payload.get("channelId", "")

    await session_manager.destroy_session(session_id)
    await conversation_history.clear(session_id)
    # 清理各模块的 session 级缓存
    interaction_graph.cleanup_session(session_id)
    search_cache.clear_session(session_id)
    # 清理 Avatar driver 和 gesture 缓存
    clear_session_driver(session_id)
    # 清理音频缓冲区
    _audio_buffers.pop(session_id, None)
    _asr_rms_buffers.pop(session_id, None)
    # 关闭讯飞流式 ASR 连接（若有）
    await xunfei_asr_manager.remove(session_id)

    await _send(websocket, {
        "type": "session.destroyed",
        "payload": {
            "sessionId": session_id,
            "channelId": channel_id,
        },
    })


async def _handle_skill_mount(payload: dict, websocket: WebSocket) -> None:
    """处理 Skill 挂载请求"""
    session_id = payload.get("sessionId", "")
    skill_id = payload.get("skillId", "")
    channel_id = payload.get("channelId", "")

    await session_manager.mount_skill(session_id, skill_id)

    await _send(websocket, {
        "type": "skill.mounted",
        "payload": {
            "sessionId": session_id,
            "skillId": skill_id,
            "channelId": channel_id,
        },
    })


async def _handle_skill_unmount(payload: dict, websocket: WebSocket) -> None:
    """处理 Skill 卸载请求"""
    session_id = payload.get("sessionId", "")
    skill_id = payload.get("skillId", "")
    channel_id = payload.get("channelId", "")

    await session_manager.unmount_skill(session_id, skill_id)

    await _send(websocket, {
        "type": "skill.unmounted",
        "payload": {
            "sessionId": session_id,
            "skillId": skill_id,
            "channelId": channel_id,
        },
    })


# 消息去重：防止 Java 多连接转发导致同一消息处理两次
_recent_messages: dict[str, float] = {}
_DEDUP_WINDOW = 1.0  # 秒（仅防重复发送，不阻止用户重复提问同一问题）


async def _handle_chat_send(payload: dict, websocket: WebSocket) -> None:
    """处理 Java 后端转发来的 chat.send 消息

    使用 asyncio.create_task 启动 workflow，不阻塞 WS 消息循环，
    以便 interrupt 等消息可以及时处理。
    """
    session_id = payload.get("sessionId", "")
    user_id = payload.get("userId", "")
    channel_id = payload.get("channelId", "")
    text = payload.get("text", "")
    skill_ids = payload.get("skillIds", [])
    asr_text = payload.get("asrText")

    # 去重：同一 session 相同文本 3 秒内只处理一次
    dedup_key = f"{session_id}:{text}"
    now = time.time()
    if dedup_key in _recent_messages and (now - _recent_messages[dedup_key]) < _DEDUP_WINDOW:
        logger.warning("重复消息已忽略: session_id=%s, text=%s", session_id, text[:30])
        return
    _recent_messages[dedup_key] = now
    # 清理过期条目（避免内存泄漏）
    expired = [k for k, t in _recent_messages.items() if now - t > 5.0]
    for k in expired:
        del _recent_messages[k]
    # 场景与设备信息：优先从 payload 获取，其次从 session 读取
    scene_id = payload.get("sceneId", "")
    device_id = payload.get("deviceId", "")
    device_location = payload.get("deviceLocation", "")
    user_type = payload.get("userType", "general")
    # 语音路径会带上 ASR 耗时（毫秒），用于分阶段 [TIMING] 汇总
    asr_ms = payload.get("asrMs")

    if not scene_id or not device_id or not user_type:
        session = await session_manager.get_session(session_id)
        if session:
            scene_id = scene_id or session.sceneId
            device_id = device_id or session.deviceId
            device_location = device_location or session.deviceLocation
            user_type = user_type if user_type != "general" else session.userType

    logger.info("Chat 请求: session_id=%s, scene_id=%s, device_id=%s, text=%s, skill_ids=%s",
                session_id, scene_id, device_id, text[:50] if text else "", skill_ids)

    # 追加用户消息到对话历史
    await conversation_history.append(session_id, "user", text)

    # 获取 Skill 提供的工具定义
    # 前端通常只发 sceneId 不带 skillIds：scene_id 与 skill 同名，
    # 为空时从场景解析技能，保证导诊等工具可用（否则导航手势无法触发）
    if not skill_ids and scene_id and skill_loader.get_skill(scene_id):
        skill_ids = [scene_id]
    available_tools = skill_loader.get_tools_for_skills(skill_ids)

    async def _send_with_context(msg: dict) -> None:
        """在回复消息中附带 channelId/userId"""
        if "payload" not in msg:
            msg["payload"] = {}
        msg["payload"]["channelId"] = channel_id
        msg["payload"]["userId"] = user_id
        await _send(websocket, msg)

    async def _run_workflow_and_record() -> None:
        """在后台运行 workflow 并记录助手回复到对话历史"""
        try:
            # 在锁内加载最新对话历史，确保前一个 workflow 完成后读到的是最新数据
            messages = await conversation_history.get_messages(session_id)
            await interaction_graph.run(
                session_id=session_id,
                messages=messages,
                skill_ids=skill_ids,
                tools=available_tools if available_tools else None,
                send_func=_send_with_context,
                asr_text=asr_text,
                scene_id=scene_id,
                device_id=device_id,
                device_location=device_location,
                user_type=user_type,
                asr_ms=asr_ms,
            )
            # 将助手回复追加到对话历史
            reply = interaction_graph.get_last_reply(session_id)
            if reply:
                await conversation_history.append(session_id, "assistant", reply)
            interaction_graph.clear_last_reply(session_id)
        except Exception as e:
            logger.error("Workflow 异常: session_id=%s, error=%s", session_id, e)
        finally:
            await session_manager.refresh_ttl(session_id)

    # 使用 create_task 启动 workflow，不阻塞 WS 消息循环
    # 这样 interrupt 等消息可以及时处理
    asyncio.create_task(_run_workflow_and_record())


async def _handle_interrupt(payload: dict, websocket: WebSocket) -> None:
    """处理中断消息：保存讲解断点，支持用户插话后续讲
    
    Payload 可选字段：
      - currentPosition: int — 当前已播放/已显示的字符位置（由前端提供）
      - currentText: str — 当前回复文本（若前端知道）
    """
    session_id = payload.get("sessionId", "")
    channel_id = payload.get("channelId", "")
    current_position = payload.get("currentPosition", 0)
    current_text = payload.get("currentText", "")

    logger.info("中断请求: session_id=%s, position=%d", session_id, current_position)

    # 保存讲解断点（用于续讲）
    # 优先使用前端提供的位置，否则尝试从 interaction_graph 获取上次回复
    if current_text and current_position >= 0:
        breakpoint_manager.save(session_id, current_text, current_position)
    else:
        last_reply = interaction_graph.get_last_reply(session_id)
        if last_reply:
            # 没有位置信息时，假设已播放一半（保守策略）
            position = len(last_reply) // 2
            breakpoint_manager.save(session_id, last_reply, position)
            logger.info("从上次回复保存断点: session_id=%s, position=%d", session_id, position)

    await interaction_graph.interrupt(session_id)

    await _send(websocket, {
        "type": "interrupt.ack",
        "payload": {
            "sessionId": session_id,
            "channelId": channel_id,
        },
    })


async def _handle_greeting_trigger(payload: dict, websocket: WebSocket) -> None:
    """处理主动问候触发

    根据时间、场景、设备位置生成上下文相关的问候语，
    通过 TTS 合成语音并驱动数字人表情动作。
    """
    session_id = payload.get("sessionId", "")
    channel_id = payload.get("channelId", "")
    user_id = payload.get("userId", "")
    trigger_type = payload.get("triggerType", "presence")
    scene_id = payload.get("sceneId", "")
    device_id = payload.get("deviceId", "")
    device_location = payload.get("deviceLocation", "")

    logger.info("问候触发: session_id=%s, trigger=%s, scene=%s, device=%s",
                session_id, trigger_type, scene_id, device_location)

    # 前端可下发 text 覆盖（如摄像头引导语"教暗号"，含形象专属唤醒词），有则直接用
    custom_text = (payload.get("text") or "").strip()
    if custom_text:
        greeting_text = custom_text
    elif trigger_type == "wake":
        greeting_text = "你好！有什么可以帮您的吗？"
    else:
        # 人体感应无文案（前端引导语未就绪）→ 不播，避免播旧的场景问候
        logger.info("人体感应无引导文案，跳过问候: session_id=%s", session_id)
        return

    async def _send_with_context(msg: dict) -> None:
        if "payload" not in msg:
            msg["payload"] = {}
        msg["payload"]["channelId"] = channel_id
        msg["payload"]["userId"] = user_id
        await _send(websocket, msg)

    async def _run_greeting() -> None:
        logger.info("开始执行问候任务: session_id=%s, text=%s", session_id, greeting_text)
        try:
            # 1. 发送流式文本（让前端显示问候语）
            await _send_with_context({
                "type": "ai.stream",
                "payload": {
                    "sessionId": session_id,
                    "text": greeting_text,
                    "done": True,
                },
            })
            logger.info("问候文本已发送: session_id=%s", session_id)
        except Exception as e:
            logger.exception("问候文本发送失败: session_id=%s", session_id)

        try:
            # 2. TTS 合成问候语音
            audio_chunks = []
            async for chunk in tts_service.synthesize_stream(greeting_text):
                audio_chunks.append(chunk)
                await _send_with_context({
                    "type": "tts.audio",
                    "payload": {
                        "sessionId": session_id,
                        "audio": base64.b64encode(chunk).decode(),
                        "format": "pcm",
                        "index": len(audio_chunks) - 1,
                        "done": False,
                    },
                })
            # 发送 TTS 结束标记
            await _send_with_context({
                "type": "tts.audio",
                "payload": {
                    "sessionId": session_id,
                    "audio": "",
                    "format": "pcm",
                    "index": len(audio_chunks),
                    "done": True,
                },
            })
            logger.info("问候TTS已发送: session_id=%s, chunks=%d", session_id, len(audio_chunks))
        except Exception as e:
            logger.exception("问候TTS合成失败: session_id=%s", session_id)

        try:
            # 3. 驱动数字人播放问候视频
            driver = get_session_driver(session_id)
            drive_data = await driver.generate_drive("greeting", greeting_text)
            await _send_with_context({
                "type": "avatar.drive",
                "payload": {
                    "sessionId": session_id,
                    **drive_data,
                },
            })
        except Exception as e:
            logger.exception("问候Avatar驱动失败: session_id=%s", session_id)

        try:
            # 4. 记录到对话历史
            await conversation_history.append(session_id, "assistant", greeting_text)
        except Exception as e:
            logger.exception("问候历史记录失败: session_id=%s", session_id)

    asyncio.create_task(_run_greeting())


async def _handle_ai_generate(payload: dict, websocket: WebSocket) -> None:
    """处理 ai.generate 消息（兼容旧协议）"""
    session_id = payload.get("sessionId", "")
    messages = payload.get("messages", [])
    skill_ids = payload.get("skillIds", [])
    tools = payload.get("tools", [])

    await interaction_graph.run(
        session_id=session_id,
        messages=messages,
        skill_ids=skill_ids,
        tools=tools,
        send_func=lambda msg: _send(websocket, msg),
    )


async def _send_noise_reply(session_id: str, channel_id: str, user_id: str, websocket: WebSocket) -> None:
    """发送噪音固定回复（ai.stream + TTS），不经过 LLM"""
    text = NOISE_REPLY_TEXT

    async def _send_ctx(msg: dict) -> None:
        if "payload" not in msg:
            msg["payload"] = {}
        msg["payload"]["channelId"] = channel_id
        msg["payload"]["userId"] = user_id
        await _send(websocket, msg)

    # 1. 发送文本
    await _send_ctx({
        "type": "ai.stream",
        "payload": {"sessionId": session_id, "text": text, "done": True},
    })

    # 2. TTS 合成并发送音频
    try:
        audio_chunks = []
        async for chunk in tts_service.synthesize_stream(text):
            audio_chunks.append(chunk)
            await _send_ctx({
                "type": "tts.audio",
                "payload": {
                    "sessionId": session_id,
                    "audio": base64.b64encode(chunk).decode(),
                    "format": "pcm",
                    "index": len(audio_chunks) - 1,
                    "done": False,
                },
            })
        await _send_ctx({
            "type": "tts.audio",
            "payload": {
                "sessionId": session_id,
                "audio": "",
                "format": "pcm",
                "index": len(audio_chunks),
                "done": True,
            },
        })
    except Exception as e:
        logger.exception("噪音回复 TTS 合成失败: session_id=%s", session_id)

    # 3. 记录到对话历史
    try:
        await conversation_history.append(session_id, "assistant", text)
    except Exception as e:
        logger.exception("噪音回复历史记录失败: session_id=%s", session_id)


async def _handle_audio_stream(payload: dict, websocket: WebSocket) -> None:
    """处理语音分片消息：接收音频数据，缓存到会话缓冲区

    协议：
    - audio.stream: {"type": "audio.stream", "payload": {"sessionId": "...", "audio": "<base64>", "index": 0, "format": "pcm"}}
    - audio.end: {"type": "audio.end", "payload": {"sessionId": "...", "format": "pcm"}}
    """
    session_id = payload.get("sessionId", "")
    audio_b64 = payload.get("audio", "")
    index = payload.get("index", 0)

    # Base64 解码音频数据
    try:
        audio_data = base64.b64decode(audio_b64) if audio_b64 else b""
    except Exception as e:
        logger.error("音频分片解码失败: session_id=%s, error=%s", session_id, e)
        return

    # ── 音频增强：VAD + 降噪（展馆嘈杂环境优化） ──
    if settings.audio_enhance.enabled:
        enhancer = get_audio_enhancer(EnhancementConfig(
            vad_aggressiveness=settings.audio_enhance.vad_aggressiveness,
            noise_reduce_strength=settings.audio_enhance.noise_reduce_strength,
            min_speech_duration_ms=settings.audio_enhance.min_speech_duration_ms,
        ))
        audio_data = enhancer.process(audio_data)
        if not audio_data:
            # VAD 判定为噪音帧，跳过
            logger.debug("VAD 过滤噪音帧: session_id=%s, index=%d", session_id, index)
            return

    if settings.asr.provider == "xunfei":
        # 讯飞流式：边说边识别。累积 RMS 门控缓冲 + 实时喂给云端会话
        if session_id not in _asr_rms_buffers:
            _asr_rms_buffers[session_id] = bytearray()
        _asr_rms_buffers[session_id].extend(audio_data)
        try:
            async def _on_partial(text: str) -> None:
                """中间识别结果推送给客户端（实时转写反馈）"""
                await _send(websocket, {
                    "type": "asr.partial",
                    "payload": {"sessionId": session_id, "text": text},
                })

            sess = await xunfei_asr_manager.get_or_create(session_id, on_partial=_on_partial)
            await sess.feed(audio_data)
        except Exception as e:
            logger.error("讯飞 ASR 喂入失败: session_id=%s, error=%s", session_id, e)
        logger.debug("音频分片(讯飞): session_id=%s, index=%d, bytes=%d", session_id, index, len(audio_data))
        return

    # whisper 路径：攒整段，audio.end 时一次性识别
    if session_id not in _audio_buffers:
        _audio_buffers[session_id] = []
    _audio_buffers[session_id].append(audio_data)
    logger.debug("音频分片: session_id=%s, index=%d, bytes=%d", session_id, index, len(audio_data))


async def _handle_audio_end(payload: dict, websocket: WebSocket) -> None:
    """处理语音结束消息：执行 ASR 识别，然后进入对话流程。

    按 provider 分流：xunfei 走实时流式（音频已在 audio.stream 阶段喂给云端），
    whisper 走整段批量识别。识别完成后统一走噪音过滤 → chat.send。
    """
    session_id = payload.get("sessionId", "")
    user_id = payload.get("userId", "")
    channel_id = payload.get("channelId", "")
    fmt = payload.get("format", "pcm")

    logger.info("语音结束: session_id=%s, format=%s, provider=%s", session_id, fmt, settings.asr.provider)

    if settings.asr.provider == "xunfei":
        asr_text, asr_ms = await _recognize_xunfei(session_id, channel_id, user_id, websocket)
    else:
        asr_text, asr_ms = await _recognize_whisper(session_id, channel_id, user_id, websocket)

    # None 表示识别阶段已自行处理（无音频/静音门控），中止
    if asr_text is None:
        return
    if not asr_text:
        logger.info("ASR 无识别结果: session_id=%s", session_id)
        return

    # ── 黑名单：幻觉词命中 → 不送 LLM ──
    if is_hallucination(asr_text):
        logger.info("噪音黑名单命中，跳过对话: session_id=%s, text='%s'", session_id, asr_text[:40])
        await _send_noise_reply(session_id, channel_id, user_id, websocket)
        return

    # ASR 完成后，将识别文本作为 chat.send 进入对话流程
    # 从 session 获取已挂载的 skill_ids（而非硬编码为空）
    session = await session_manager.get_session(session_id)
    mounted_skills = session.mountedSkills if session else []

    await _handle_chat_send({
        "sessionId": session_id,
        "userId": user_id,
        "channelId": channel_id,
        "text": asr_text,
        "skillIds": mounted_skills,
        "asrText": asr_text,
        "asrMs": asr_ms,
    }, websocket)


async def _recognize_xunfei(
    session_id: str, channel_id: str, user_id: str, websocket: WebSocket
) -> tuple[str | None, int]:
    """讯飞流式识别收尾：RMS 门控 → finish() 取最终结果 → 推送 asr.result。

    返回 (asr_text, asr_ms)；asr_text=None 表示已发噪音回复或无音频，调用方应中止。
    """
    full_audio = bytes(_asr_rms_buffers.pop(session_id, bytearray()))
    sess = xunfei_asr_manager.get(session_id)

    if not full_audio or sess is None:
        logger.warning("讯飞 ASR 无音频/无会话: session_id=%s", session_id)
        await xunfei_asr_manager.remove(session_id)
        return None, 0

    # ── L1 VAD 门控：RMS 过低视为静音/底噪，不取识别结果 ──
    rms = compute_rms(full_audio)
    if rms < settings.rag.audio_rms_threshold:
        logger.info("L1 VAD 门控: RMS=%.5f < %.5f，判定为静音/底噪: session_id=%s",
                    rms, settings.rag.audio_rms_threshold, session_id)
        await xunfei_asr_manager.remove(session_id)
        await _send_noise_reply(session_id, channel_id, user_id, websocket)
        return None, 0

    # 音频已在 audio.stream 阶段实时喂入，这里只需收尾等待最终结果。
    # 计时从 finish() 起算：衡量"说完→拿到最终文本"的尾延迟（流式下应远小于整段识别）。
    t_asr_start = time.monotonic()
    asr_text = await sess.finish()
    asr_ms = int((time.monotonic() - t_asr_start) * 1000)
    await xunfei_asr_manager.remove(session_id)

    logger.info("[TIMING] ASR 识别耗时(讯飞流式): session_id=%s, asr=%dms, text='%s'",
                session_id, asr_ms, (asr_text or "")[:30])

    await _send(websocket, {
        "type": "asr.result",
        "payload": {
            "sessionId": session_id,
            "channelId": channel_id,
            "userId": user_id,
            "text": asr_text,
            "isFinal": True,
            "asrMs": asr_ms,
        },
    })
    return asr_text, asr_ms


async def _recognize_whisper(
    session_id: str, channel_id: str, user_id: str, websocket: WebSocket
) -> tuple[str | None, int]:
    """Whisper 整段识别：合并分片 → RMS 门控 → 批量识别 → 推送 asr.result。

    返回 (asr_text, asr_ms)；asr_text=None 表示已发噪音回复或无音频，调用方应中止。
    """
    chunks = _audio_buffers.pop(session_id, [])
    if not chunks:
        logger.warning("无音频数据: session_id=%s", session_id)
        return None, 0

    full_audio = b"".join(chunks)
    logger.info("合并音频: session_id=%s, chunks=%d, total_bytes=%d", session_id, len(chunks), len(full_audio))

    # ── L1 VAD 门控：RMS 过低视为静音/底噪，不送 ASR ──
    rms = compute_rms(full_audio)
    if rms < settings.rag.audio_rms_threshold:
        logger.info("L1 VAD 门控: RMS=%.5f < %.5f，判定为静音/底噪，跳过 ASR: session_id=%s",
                    rms, settings.rag.audio_rms_threshold, session_id)
        await _send_noise_reply(session_id, channel_id, user_id, websocket)
        return None, 0

    async def asr_send_func(msg: dict) -> None:
        if "payload" not in msg:
            msg["payload"] = {}
        msg["payload"]["channelId"] = channel_id
        msg["payload"]["userId"] = user_id
        await _send(websocket, msg)

    async def audio_chunk_generator():
        yield full_audio

    t_asr_start = time.monotonic()
    asr_text = await asr_service.transcribe_stream(
        audio_chunk_generator(),
        session_id=session_id,
        send_func=asr_send_func,
    )
    asr_ms = int((time.monotonic() - t_asr_start) * 1000)
    logger.info("[TIMING] ASR 识别耗时(whisper): session_id=%s, asr=%dms, text='%s'",
                session_id, asr_ms, (asr_text or "")[:30])
    return asr_text, asr_ms


async def _handle_tts_request(payload: dict, websocket: WebSocket) -> None:
    """处理 TTS 请求：流式合成音频并回传"""
    session_id = payload.get("sessionId", "")
    channel_id = payload.get("channelId", "")
    user_id = payload.get("userId", "")
    text = payload.get("text", "")
    voice = payload.get("voice", "default")
    fmt = payload.get("format", "raw")

    logger.info("TTS 请求: session_id=%s, text_len=%d", session_id, len(text))

    chunk_index = 0
    async for audio_chunk in tts_service.synthesize_stream(text, voice):
        if audio_chunk:
            # 将音频数据 base64 编码回传
            audio_b64 = base64.b64encode(audio_chunk).decode("ascii")
            await _send(websocket, {
                "type": "tts.audio",
                "payload": {
                    "sessionId": session_id,
                    "channelId": channel_id,
                    "userId": user_id,
                    "audio": audio_b64,
                    "index": chunk_index,
                    "format": "pcm",
                    "done": False,
                },
            })
            chunk_index += 1

    # 发送完成标记
    await _send(websocket, {
        "type": "tts.audio",
        "payload": {
            "sessionId": session_id,
            "channelId": channel_id,
            "userId": user_id,
            "audio": "",
            "index": chunk_index,
            "format": "pcm",
            "done": True,
        },
    })


async def _handle_asr_request(payload: dict, websocket: WebSocket) -> None:
    session_id = payload.get("sessionId", "")
    channel_id = payload.get("channelId", "")
    user_id = payload.get("userId", "")
    audio = payload.get("audio", b"")
    fmt = payload.get("format", "wav")
    # source 由前端标注请求来源（如 'wake' 表示唤醒词检测），原样回显以便前端区分处理
    source = payload.get("source", "")

    # 如果 audio 是 base64 字符串，解码
    if isinstance(audio, str) and audio:
        try:
            audio = base64.b64decode(audio)
        except Exception:
            pass

    logger.info("ASR 请求: session_id=%s, audio_bytes=%d, source=%s",
                session_id, len(audio) if isinstance(audio, (bytes, bytearray)) else 0, source)

    # 唤醒请求：讯飞云端 ASR 转写 + pypinyin 模糊匹配
    # （KWS 3.3M 本地模型对真实麦克风音频完全无响应，已弃用；云端 ASR 抗噪强、准确率高）
    is_wake = source == "wake"
    wake_phrases = payload.get("wakePhrases") or []

    if isinstance(audio, bytes) and audio:

        if is_wake:
            # ── 唤醒检测：根据 ASR provider 分流 ──
            incoming_rate = int(payload.get("sampleRate") or 16000)
            hotwords = "|".join(wake_phrases) if wake_phrases else None

            if settings.asr.provider == "xunfei":
                # 讯飞云端 ASR 唤醒检测：转写 + pypinyin 模糊匹配
                # 重采样到 16kHz（讯飞要求 16k/16bit/单声道）
                if incoming_rate != 16000:
                    import numpy as _np
                    _samples = _np.frombuffer(audio, dtype=_np.int16).astype(_np.float32) / 32768.0
                    _dur = len(_samples) / incoming_rate
                    _target_len = int(_dur * 16000)
                    _x_old = _np.linspace(0, _dur, len(_samples), endpoint=False)
                    _x_new = _np.linspace(0, _dur, _target_len, endpoint=False)
                    _resampled = _np.interp(_x_new, _x_old, _samples)
                    audio = (_resampled * 32767).astype(_np.int16).tobytes()

                t_asr_start = time.monotonic()
                wake_session = XunfeiAsrSession(hotwords=hotwords)
                try:
                    await wake_session.start()
                    await wake_session.feed(audio)
                    text = await wake_session.finish(timeout=5.0)
                except Exception as e:
                    logger.error("讯飞唤醒 ASR 异常，降级 Whisper: %s", e)
                    text = ""
                finally:
                    await wake_session.close()
                # 讯飞失败或返回空 → 降级 Whisper 本地
                if not text:
                    wav_audio = asr_service._pcm_to_wav(audio, sample_rate=16000)
                    text = await asr_service.transcribe(wav_audio, format="wav", wake=True, initial_prompt=hotwords)
                asr_ms = int((time.monotonic() - t_asr_start) * 1000)
                matched = match_wake(text, wake_phrases) if text and wake_phrases else False
            else:
                # Whisper 本地唤醒检测（走 asr_service.transcribe，wake=True 用小模型）
                # 如果前端送的是 PCM 未加 WAV 头，需要转换
                if fmt == "pcm" or incoming_rate > 0:
                    wav_audio = asr_service._pcm_to_wav(audio, sample_rate=incoming_rate if incoming_rate > 0 else 16000)
                else:
                    wav_audio = audio
                t_asr_start = time.monotonic()
                text = await asr_service.transcribe(wav_audio, fmt="wav", wake=True, initial_prompt=hotwords)
                asr_ms = int((time.monotonic() - t_asr_start) * 1000)
                matched = match_wake(text, wake_phrases) if text and wake_phrases else False

            logger.info("[TIMING] 唤醒检测: session_id=%s, provider=%s, asr=%dms, text=%s, matched=%s",
                        session_id, settings.asr.provider, asr_ms, (text or "")[:30], matched)
            resp_payload = {
                "sessionId": session_id,
                "channelId": channel_id,
                "userId": user_id,
                "text": text or "",
                "source": source,
                "asrMs": asr_ms,
                "matched": matched,
            }
            await _send(websocket, {"type": "asr.result", "payload": resp_payload})
        else:
            # ── 非唤醒：常规 ASR 转写 ──
            t_asr_start = time.monotonic()
            result = await asr_service.transcribe(audio, fmt, wake=False)
            asr_ms = int((time.monotonic() - t_asr_start) * 1000)
            logger.info("[TIMING] ASR 识别耗时: session_id=%s, source=%s, asr=%dms, text='%s'",
                        session_id, source or "-", asr_ms, (result or "")[:30])
            resp_payload = {
                "sessionId": session_id,
                "channelId": channel_id,
                "userId": user_id,
                "text": result,
                "source": source,
                "asrMs": asr_ms,
            }
            await _send(websocket, {"type": "asr.result", "payload": resp_payload})
    else:
        logger.warning("ASR 请求无音频数据: session_id=%s", session_id)


async def _handle_mcp_tool_call(payload: dict, request_id: str | None, websocket: WebSocket) -> None:
    tool_name = payload.get("toolName", "")
    arguments = payload.get("arguments", {})
    context = payload.get("context")

    try:
        tool = tool_registry.get_tool(tool_name)
        if tool is None or tool.handler is None:
            raise ValueError(f"工具不存在或无处理器: {tool_name}")

        result = await sandbox_executor.execute(
            handler=tool.handler,
            arguments=arguments,
            tool_name=tool_name,
            permissions=tool.permissions,
        )
        logger.info("MCP 工具调用成功: tool=%s, result=%s", tool_name, result)
        await _send(websocket, {
            "type": "mcp.tool.result",
            "payload": {"requestId": request_id, "result": result},
        })
    except Exception as e:
        logger.error("MCP 工具调用失败: tool=%s, error=%s", tool_name, e)
        await _send(websocket, {
            "type": "error",
            "payload": {"code": 2004, "message": f"工具 {tool_name} 执行失败: {e}"},
        })


async def cleanup_connection_sessions(session_ids: set[str]) -> None:
    """WS 断连时清理残留资源（音频缓冲/ASR 会话）

    当客户端未显式发送 session.destroy 就断连时（如网络中断、浏览器关闭），
    由此函数清理该连接上所有 session 的进程级缓存，防止内存泄漏。
    注意：不清理 Redis 中的 session/历史（它们有 TTL 自动过期）。
    """
    for sid in session_ids:
        _audio_buffers.pop(sid, None)
        _asr_rms_buffers.pop(sid, None)
        clear_session_driver(sid)
        try:
            await xunfei_asr_manager.remove(sid)
        except Exception:
            pass
