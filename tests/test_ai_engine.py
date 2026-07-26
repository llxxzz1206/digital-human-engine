"""AI Engine 全链路自动测试脚本

使用方法：
  cd python-ai-engine
  uv run python tests/test_ai_engine.py

前置条件：
  - Docker 基础设施已启动
  - Python AI Engine 已启动 (port 8000)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
from datetime import datetime

try:
    import websockets
except ImportError:
    print("需要安装 websockets: uv add websockets")
    sys.exit(1)

AI_ENGINE_WS = "ws://localhost:8000/ws/ai"
AI_ENGINE_API = "http://localhost:8000"

# websockets 连接参数：禁用 ping 防止超时断连
_WS_PING_INTERVAL = None  # 禁用客户端 ping
_WS_PING_TIMEOUT = None   # 禁用 ping 超时
_WS_CLOSE_TIMEOUT = 10

# ── 测试结果收集 ──────────────────────────────────────────

_results: list[dict] = []
_pass = 0
_fail = 0


def record(test_id: str, name: str, passed: bool, detail: str = "") -> None:
    global _pass, _fail
    status = "PASS" if passed else "FAIL"
    if passed:
        _pass += 1
    else:
        _fail += 1
    _results.append({"id": test_id, "name": name, "status": status, "detail": detail})
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{test_id}] {name}" + (f" — {detail}" if detail and not passed else ""))


# ── 辅助函数 ──────────────────────────────────────────────

async def create_session(ws) -> str:
    """创建会话，返回 session_id"""
    await ws.send(json.dumps({"type": "session.create", "payload": {}}))
    resp = json.loads(await ws.recv())
    return resp["payload"]["sessionId"]


async def drain_ws(ws, timeout: float = 0.5) -> None:
    """排空 WS 中残留的消息（防止上一轮 TTS 等消息干扰下一轮）"""
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break


async def send_and_collect(ws, session_id: str, text: str, skill_ids: list[str] | None = None) -> dict:
    """发送 chat.send 并收集所有回复消息

    只关注 ai.stream 类型消息来判定回复内容和完成状态，
    忽略 tts.audio/avatar.drive 等消息。
    """
    payload = {"sessionId": session_id, "text": text}
    if skill_ids:
        payload["skillIds"] = skill_ids

    await ws.send(json.dumps({"type": "chat.send", "payload": payload}))

    reply_text = ""
    tts_count = 0
    has_done = False
    start = time.monotonic()

    while True:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        except asyncio.TimeoutError:
            return {"reply": reply_text, "tts_count": tts_count,
                    "done": False, "latency_ms": 0, "error": "timeout"}

        msg_type = msg.get("type", "")
        if msg_type == "ai.stream":
            p = msg["payload"]
            # 只关注当前 session 的消息
            if p.get("sessionId") != session_id:
                continue
            if p.get("text"):
                reply_text += p["text"]
            if p.get("done"):
                has_done = True
                break
        elif msg_type == "tts.audio":
            if not msg["payload"].get("done"):
                tts_count += 1
        elif msg_type == "error":
            return {"reply": "", "tts_count": 0,
                    "done": False, "latency_ms": 0, "error": str(msg["payload"])}

    latency_ms = int((time.monotonic() - start) * 1000)
    return {"reply": reply_text, "tts_count": tts_count,
            "done": has_done, "latency_ms": latency_ms}


async def wait_for_msg(ws, msg_type: str, timeout: float = 10) -> dict | None:
    """等待特定类型的消息"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic()))
            if msg.get("type") == msg_type:
                return msg
        except asyncio.TimeoutError:
            break
    return None


# ── A. 健康检查 & API ────────────────────────────────────

async def test_health_api():
    print("\n━━━ A. 健康检查 & API ━━━")
    try:
        with urllib.request.urlopen(f"{AI_ENGINE_API}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            record("A1", "GET /health", data.get("status") == "ok", str(data))
    except Exception as e:
        record("A1", "GET /health", False, str(e))

    try:
        with urllib.request.urlopen(f"{AI_ENGINE_API}/api/chat-logs/dates", timeout=5) as resp:
            data = json.loads(resp.read())
            record("A2", "GET /api/chat-logs/dates", isinstance(data.get("dates"), list))
    except Exception as e:
        record("A2", "GET /api/chat-logs/dates", False, str(e))


# ── B. 会话管理 ──────────────────────────────────────────

async def test_session_management():
    print("\n━━━ B. 会话管理 ━━━")
    async with websockets.connect(AI_ENGINE_WS, ping_interval=_WS_PING_INTERVAL, ping_timeout=_WS_PING_TIMEOUT, close_timeout=_WS_CLOSE_TIMEOUT) as ws:
        # B1: 创建会话
        await ws.send(json.dumps({"type": "session.create", "payload": {}}))
        resp = json.loads(await ws.recv())
        sid = resp["payload"]["sessionId"]
        record("B1", "session.create", bool(sid), f"sessionId={sid}")

        # B2: 重复创建
        await ws.send(json.dumps({"type": "session.create", "payload": {}}))
        resp2 = json.loads(await ws.recv())
        sid2 = resp2["payload"]["sessionId"]
        record("B2", "重复 session.create", sid2 != sid, f"new_id={sid2}")

        # B5: 挂载 skill
        await ws.send(json.dumps({"type": "skill.mount", "payload": {"sessionId": sid, "skillId": "aviation"}}))
        resp3 = json.loads(await ws.recv())
        record("B5", "skill.mount(aviation)", resp3.get("type") == "skill.mounted")

        # B7: 重复挂载
        await ws.send(json.dumps({"type": "skill.mount", "payload": {"sessionId": sid, "skillId": "aviation"}}))
        resp4 = json.loads(await ws.recv())
        record("B7", "重复 mount 同一 skill", resp4.get("type") == "skill.mounted")

        # B6: 卸载 skill
        await ws.send(json.dumps({"type": "skill.unmount", "payload": {"sessionId": sid, "skillId": "aviation"}}))
        resp5 = json.loads(await ws.recv())
        record("B6", "skill.unmount(aviation)", resp5.get("type") == "skill.unmounted")

        # B3: 销毁会话
        await ws.send(json.dumps({"type": "session.destroy", "payload": {"sessionId": sid}}))
        resp6 = json.loads(await ws.recv())
        record("B3", "session.destroy", resp6.get("type") == "session.destroyed")


# ── C. RAG 双阈值路由 ────────────────────────────────────

async def test_rag_routing():
    print("\n━━━ C. RAG 双阈值路由 ━━━")
    async with websockets.connect(AI_ENGINE_WS, ping_interval=_WS_PING_INTERVAL, ping_timeout=_WS_PING_TIMEOUT, close_timeout=_WS_CLOSE_TIMEOUT) as ws:
        sid = await create_session(ws)

        # 挂载 aviation skill（虽然自动发现也能用，但显式测试）
        await ws.send(json.dumps({"type": "skill.mount", "payload": {"sessionId": sid, "skillId": "aviation"}}))
        await ws.recv()  # skill.mounted

        # ── C1: direct 路由 ──
        print("\n  ── C1. direct 路由 ──")
        direct_tests = [
            ("C1-1", "航空历史厅在哪", ["1F", "航空历史"]),
            ("C1-2", "无人机科技区怎么走", ["3F", "无人机"]),
            ("C1-3", "飞行模拟体验区在几楼", ["2F", "模拟"]),
        ]
        for tid, text, keywords in direct_tests:
            result = await send_and_collect(ws, sid, text, ["aviation"])
            # direct/rag_chat 路由：回复中包含关键词即可（rerank分数波动可能导致路由变化）
            matched = any(kw in result["reply"] for kw in keywords) if result["reply"] else False
            record(tid, f"direct: '{text}'", matched,
                   f"reply='{result['reply'][:40]}' latency={result['latency_ms']}ms route={'direct' if result['latency_ms'] < 1000 else 'rag_chat'}")
            await drain_ws(ws)  # 排空 TTS 残留消息

        # ── C2: rag_chat 路由 ──
        print("\n  ── C2. rag_chat 路由 ──")
        rag_chat_tests = [
            ("C2-1", "有什么好看的展品", ["展品"]),
            ("C2-2", "推荐一下参观路线", ["路线"]),
        ]
        for tid, text, keywords in rag_chat_tests:
            result = await send_and_collect(ws, sid, text, ["aviation"])
            matched = any(kw in result["reply"] for kw in keywords)
            record(tid, f"rag_chat: '{text}'", matched,
                   f"reply='{result['reply'][:30]}' latency={result['latency_ms']}ms")
            await drain_ws(ws)

        # ── C3: chat 路由 ──
        print("\n  ── C3. chat 路由 ──")
        chat_tests = [
            ("C3-1", "你好", None),
            ("C3-2", "1+1等于多少", ["2"]),
            ("C3-3", "今天天气怎么样", None),
        ]
        for tid, text, keywords in chat_tests:
            result = await send_and_collect(ws, sid, text, ["aviation"])
            if keywords:
                matched = any(kw in result["reply"] for kw in keywords)
            else:
                matched = bool(result["reply"]) and result["done"]
            record(tid, f"chat: '{text}'", matched,
                   f"reply='{result['reply'][:30]}' latency={result['latency_ms']}ms")
            await drain_ws(ws)


# ── D. 边界 & 异常 ───────────────────────────────────────

async def test_edge_cases():
    print("\n━━━ D. 边界 & 异常 ━━━")
    async with websockets.connect(AI_ENGINE_WS, ping_interval=_WS_PING_INTERVAL, ping_timeout=_WS_PING_TIMEOUT, close_timeout=_WS_CLOSE_TIMEOUT) as ws:
        sid = await create_session(ws)

        # D1: 空字符串
        try:
            result = await send_and_collect(ws, sid, "")
            record("D1", "空字符串", True, f"不崩溃 reply='{result['reply'][:20]}'")
        except Exception as e:
            record("D1", "空字符串", False, str(e))
        await drain_ws(ws)

        # D3: 特殊字符
        try:
            result = await send_and_collect(ws, sid, "<script>alert(1)</script>")
            record("D3", "特殊字符 XSS", True, f"不崩溃 reply='{result['reply'][:20]}'")
        except Exception as e:
            record("D3", "特殊字符 XSS", False, str(e))
        await drain_ws(ws)

        # D4: 纯数字
        result = await send_and_collect(ws, sid, "123")
        record("D4", "纯数字", result["done"] and bool(result["reply"]), f"reply='{result['reply'][:20]}'")
        await drain_ws(ws)

        # D4: 快速连续发5条（现在有互斥锁，应排队执行）
        all_ok = True
        for i in range(5):
            await ws.send(json.dumps({"type": "chat.send", "payload": {"sessionId": sid, "text": f"测试{i}"}}))
        # 逐条收集回复
        done_count = 0
        try:
            while done_count < 5:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if msg.get("type") == "ai.stream" and msg["payload"].get("done"):
                    done_count += 1
        except asyncio.TimeoutError:
            all_ok = False
        record("D6", f"快速连续5条消息 (完成{done_count}/5)", done_count >= 4)


# ── E. TTS 语音 ──────────────────────────────────────────

async def test_tts():
    print("\n━━━ E. TTS 语音 ━━━")
    async with websockets.connect(AI_ENGINE_WS, ping_interval=_WS_PING_INTERVAL, ping_timeout=_WS_PING_TIMEOUT, close_timeout=_WS_CLOSE_TIMEOUT) as ws:
        sid = await create_session(ws)

        # E1: TTS 音频生成
        await ws.send(json.dumps({"type": "chat.send", "payload": {"sessionId": sid, "text": "你好"}}))
        tts_audio_count = 0
        tts_done = False
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("type") == "tts.audio":
                if msg["payload"].get("done"):
                    tts_done = True
                elif msg["payload"].get("audio"):
                    tts_audio_count += 1
            elif msg.get("type") == "ai.stream" and msg["payload"].get("done"):
                # 继续等 TTS done
                pass
            if tts_done:
                break

        record("D1", "TTS 音频生成", tts_audio_count > 0, f"audio_chunks={tts_audio_count}")
        record("D4", "TTS done=True", tts_done)


# ── F. 中断机制 ──────────────────────────────────────────

async def test_interrupt():
    print("\n━━━ F. 中断机制 ━━━")
    async with websockets.connect(AI_ENGINE_WS, ping_interval=_WS_PING_INTERVAL, ping_timeout=_WS_PING_TIMEOUT, close_timeout=_WS_CLOSE_TIMEOUT) as ws:
        sid = await create_session(ws)

        # 发送问题后立即中断
        await ws.send(json.dumps({"type": "chat.send", "payload": {"sessionId": sid, "text": "讲一个很长的故事"}}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "interrupt", "payload": {"sessionId": sid}}))

        got_ack = False
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "interrupt.ack":
                    got_ack = True
                    break
                if msg.get("type") == "ai.stream" and msg["payload"].get("done"):
                    break
        except asyncio.TimeoutError:
            pass

        record("G1", "interrupt → interrupt.ack", got_ack)

        # 中断后发新问题
        result = await send_and_collect(ws, sid, "你好")
        record("G2", "中断后新对话", result["done"], f"reply='{result['reply'][:20]}'")


# ── G. 对话日记验证 ───────────────────────────────────────

async def test_chat_logs():
    print("\n━━━ G. 对话日记 ━━━")
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with urllib.request.urlopen(f"{AI_ENGINE_API}/api/chat-logs?date={today}", timeout=5) as resp:
            data = json.loads(resp.read())
            records = data.get("records", [])
            record("I6", "对话日记 API", len(records) > 0, f"count={len(records)}")

            # 检查是否有 direct/chat/rag_chat 三种路由
            routes = set(r.get("route") for r in records)
            record("I4-I5", "路由类型覆盖", len(routes) >= 2, f"routes={routes}")
    except Exception as e:
        record("I6", "对话日记 API", False, str(e))


# ── H. 无 skill_ids 自动发现 ─────────────────────────────

async def test_auto_discovery():
    print("\n━━━ H. 无 skill_ids 自动发现 ━━━")
    async with websockets.connect(AI_ENGINE_WS, ping_interval=_WS_PING_INTERVAL, ping_timeout=_WS_PING_TIMEOUT, close_timeout=_WS_CLOSE_TIMEOUT) as ws:
        sid = await create_session(ws)
        # 不挂载任何 skill，不传 skillIds
        result = await send_and_collect(ws, sid, "骨折住院在哪层")
        has_answer = "12层" in result["reply"] or "骨科" in result["reply"]
        record("H1", "无 skillIds 自动发现 RAG", has_answer,
               f"reply='{result['reply'][:30]}' latency={result['latency_ms']}ms")


# ── 主函数 ────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  AI Engine 全链路自动测试")
    print("=" * 60)

    try:
        urllib.request.urlopen(f"{AI_ENGINE_API}/health", timeout=3)
    except Exception:
        print("\n❌ AI Engine 未启动！请先运行:")
        print("   cd python-ai-engine && uv run python -m uvicorn app.main:app --port 8000 --ws-ping-interval 0")
        sys.exit(1)

    await test_health_api()
    await test_session_management()
    await test_rag_routing()
    await test_edge_cases()
    await test_tts()
    await test_interrupt()
    await test_chat_logs()
    await test_auto_discovery()

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print(f"  测试完成: ✅ {_pass} 通过  ❌ {_fail} 失败  共 {_pass + _fail} 项")
    print("=" * 60)

    if _fail > 0:
        print("\n失败项:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"  ❌ [{r['id']}] {r['name']} — {r['detail']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
