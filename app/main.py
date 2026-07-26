from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# 抑制 asyncio 底层 socket 错误日志（连接断开时的 send 异常）
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.api.routes import health, tools, chat_logs, faq, knowledge, admin, voice
from app.infrastructure.redis import RedisPool
from app.infrastructure.database import DatabasePool
from app.ws.message_handler import handle_message
from app.skill.loader import skill_loader

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("=== Digital Human AI Engine 启动 ===")

    # 初始化 Redis 连接池
    redis_ok = await RedisPool.ping()
    if redis_ok:
        logger.info("Redis 连接成功")
    else:
        logger.warning("Redis 连接失败，将使用内存回退模式")

    # 初始化 PostgreSQL 连接池
    db_ok = await DatabasePool.ping()
    if db_ok:
        logger.info("PostgreSQL 连接成功")
        # 确保 TTS 缓存表存在
        from app.rag.tts_cache import tts_cache
        await tts_cache.ensure_table()
        # 确保 FAQ 候选表存在
        from app.rag.faq_promotion import faq_promotion_service
        await faq_promotion_service.ensure_table()
        # 后台预热三句固定话术（填充语/好啦/收尾语），不阻塞启动
        asyncio.create_task(voice.warm_fixed_phrases())
        # 启动 TTS 预缓存任务（常用语预合成，降低首字延迟）
        from app.voice.tts_pre_cache import start_pre_cache_task
        start_pre_cache_task()
        logger.info("TTS 预缓存任务已启动")
    else:
        logger.warning("PostgreSQL 连接失败，TTS 缓存和 FAQ 功能不可用")

    # 加载 Skill 插件（代码定义）
    skills_dir = Path(__file__).parent / "skill" / "skills"
    skill_loader.load_skills(skills_dir)
    # 加载 Redis 中手动配置的 Skill
    await skill_loader.load_redis_skills()
    logger.info("Skills 已加载: %s", [s.name for s in skill_loader.list_skills()])

    # 把技能注册进 Redis 注册表（管理后台"skill管理"Tab 读取该表）
    # 仅当 key 不存在时写入，避免覆盖后台手动切换过的启用状态
    if redis_ok:
        try:
            r = await RedisPool.get()
            for s in skill_loader.list_skills():
                key = admin.SKILL_PREFIX + s.name
                if not await r.exists(key):
                    await r.hset(key, mapping={
                        "id": s.name,
                        "name": s.name,
                        "description": s.description or s.name,
                        "status": "ENABLE",
                        "knowledgeStatus": "READY" if s.knowledge_collection else "NONE",
                    })
        except Exception as e:
            logger.warning("技能注册表写入 Redis 失败（不影响运行）: %s", e)

    # 注册内置 MCP 工具
    import app.mcp.builtin_tools  # noqa: F401
    logger.info("内置 MCP 工具已注册")

    yield

    # 关闭连接池
    await DatabasePool.close()
    await RedisPool.close()

    logger.info("=== Digital Human AI Engine 关闭 ===")


app = FastAPI(
    title="Digital Human AI Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(tools.router, tags=["tools"])
app.include_router(chat_logs.router, tags=["chat-logs"])
app.include_router(faq.router, tags=["faq"])
app.include_router(knowledge.router, tags=["knowledge"])
app.include_router(admin.router, tags=["admin"])
app.include_router(voice.router, tags=["voice"])


@app.websocket("/ws")
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket, token: str | None = None):
    """前端大屏直连此 WS 端点，发送 chat.send / audio.stream 等请求

    FastAPI/Starlette 会自动响应 WebSocket Ping 帧（RFC 6455），
    无需手动处理。
    
    安全：如果 WEBSOCKET_TOKEN 配置非空，则验证 token 参数
    """
    # Token 验证
    if settings.websocket_token:
        if token != settings.websocket_token:
            await websocket.close(code=4001, reason="Invalid token")
            logger.warning("WebSocket 连接被拒绝：token 无效")
            return
    else:
        logger.warning("websocket_token 未配置，WS 处于无鉴权状态（仅限开发环境）")

    # 连接频率限制（同 IP 10次/分钟）
    client_ip = websocket.client.host if websocket.client else "unknown"
    import time as _time
    now = _time.time()
    if not hasattr(websocket_endpoint, "_conn_log"):
        websocket_endpoint._conn_log = {}  # type: ignore
    conn_log: dict = websocket_endpoint._conn_log  # type: ignore
    timestamps = conn_log.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < 60]
    if len(timestamps) >= 10:
        await websocket.close(code=4002, reason="Rate limit exceeded")
        logger.warning("WebSocket 连接被拒绝：频率超限 ip=%s", client_ip)
        return
    timestamps.append(now)
    conn_log[client_ip] = timestamps

    await websocket.accept()
    logger.info("前端 WS 已连接: %s", websocket.url.path)

    # 追踪此连接上的 session_id，断连时清理残留资源
    from app.ws.message_handler import cleanup_connection_sessions
    connection_sessions: set[str] = set()

    try:
        while True:
            data = await websocket.receive_json()
            # 追踪 session_id（从 payload 中提取）
            sid = (data.get("payload") or {}).get("sessionId", "")
            if sid:
                connection_sessions.add(sid)
            await handle_message(data, websocket)
    except WebSocketDisconnect:
        logger.info("前端 WS 已断开")
    except Exception as e:
        logger.warning("WS 异常断开: %s", e)
    finally:
        # 清理此连接上所有 session 的残留资源（音频缓冲/ASR 会话）
        if connection_sessions:
            await cleanup_connection_sessions(connection_sessions)
            logger.info("WS 断连清理: sessions=%s", connection_sessions)
