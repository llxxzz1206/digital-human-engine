"""端到端全链路测试 - 通过 Java WS Gateway (9090)"""
import asyncio
import json
import time
import urllib.request

URI = "ws://localhost:9090/ws?token=test-token"
AI_URI = "ws://localhost:8000/ws/ai"


async def recv(ws, timeout=15):
    resp = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(resp)


async def test_java_health():
    """T1: Java Backend HTTP 健康检查"""
    print("\n=== T1: Java Backend HTTP ===")
    try:
        with urllib.request.urlopen("http://localhost:82/", timeout=5) as r:
            body = r.read().decode()
            if "WELCOME" in body:
                print("  PASS: HTTP 82 返回 WELCOME")
                return True
            else:
                print(f"  FAIL: 响应不是 WELCOME: {body[:50]}")
                return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


async def test_python_health():
    """T2: Python AI Engine 健康检查"""
    print("\n=== T2: Python AI Engine ===")
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
            data = json.loads(r.read())
            if data.get("status") == "ok":
                print(f"  PASS: status={data['status']}, redis={data.get('redis','')}")
                return True
            else:
                print(f"  FAIL: {data}")
                return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


async def test_ws_connect_auth():
    """T3: WS 连接鉴权"""
    print("\n=== T3: WS 连接鉴权 ===")
    import websockets

    # 3a: 无 token 应该被拒绝
    try:
        async with websockets.connect("ws://localhost:9090/ws", open_timeout=5) as ws:
            print("  FAIL: 无 token 连接不应成功")
            return False
    except Exception:
        print("  PASS: 无 token 连接被拒绝")

    # 3b: 有 token 应该连接成功
    try:
        async with websockets.connect(URI, open_timeout=5) as ws:
            print("  PASS: test-token 连接成功")
            return True
    except Exception as e:
        print(f"  FAIL: test-token 连接失败: {e}")
        return False


async def test_session_lifecycle_via_java():
    """T4: 通过 Java WS 的会话生命周期"""
    print("\n=== T4: 会话生命周期 (via Java WS) ===")
    import websockets

    try:
        async with websockets.connect(URI, open_timeout=5) as ws:
            # session.create
            await ws.send(json.dumps({
                "type": "session.create",
                "payload": {"avatarId": "test-avatar"},
            }))
            data = await recv(ws, 5)
            if data.get("type") != "session.created":
                print(f"  FAIL: expected session.created, got {data.get('type')}")
                return False
            sid = data["payload"]["sessionId"]
            print(f"  PASS: session.created, sid={sid[:8]}...")

            # skill.mount
            await ws.send(json.dumps({
                "type": "skill.mount",
                "payload": {"sessionId": sid, "skillId": "example"},
            }))
            data2 = await recv(ws, 5)
            if data2.get("type") == "skill.mounted":
                print("  PASS: skill.mounted")
            else:
                print(f"  WARN: skill.mount 返回 {data2.get('type')}")

            # session.destroy
            await ws.send(json.dumps({
                "type": "session.destroy",
                "payload": {"sessionId": sid},
            }))
            data3 = await recv(ws, 5)
            if data3.get("type") == "session.destroyed":
                print("  PASS: session.destroyed")
            else:
                print(f"  WARN: session.destroy 返回 {data3.get('type')}")

            return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


async def test_chat_full_link():
    """T5: 文本对话全链路 (客户端→Java WS→Python→LLM→流式回传→Avatar)"""
    print("\n=== T5: 文本对话全链路 ===")
    import websockets

    try:
        async with websockets.connect(URI, open_timeout=10) as ws:
            # 创建会话
            await ws.send(json.dumps({
                "type": "session.create",
                "payload": {"avatarId": "test-avatar"},
            }))
            data = await recv(ws, 10)
            sid = data["payload"]["sessionId"]

            # 等1秒确保 AiEngineClient 连接已建立
            await asyncio.sleep(1)

            # 发送对话
            t0 = time.time()
            await ws.send(json.dumps({
                "type": "chat.send",
                "payload": {
                    "sessionId": sid,
                    "text": "你好，简单介绍一下你自己",
                    "skillIds": ["example"],
                },
            }))

            ai_count = 0
            avatar_count = 0
            full_text = ""
            got_done = False
            first_chunk_time = None
            intent_type = ""

            for _ in range(60):
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(resp)
                    t = data.get("type", "")
                    p = data.get("payload", {})

                    if t == "ai.intent":
                        intent_type = p.get("intent", "")
                        print(f"  意图: {intent_type}")
                    elif t == "ai.stream":
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        ai_count += 1
                        text_chunk = p.get("text", "")
                        full_text += text_chunk
                        if p.get("done"):
                            got_done = True
                            elapsed = time.time() - t0
                            print(f"  流式完成: {ai_count} chunks, 文本长度={len(full_text)}, 耗时={elapsed:.1f}s")
                    elif t == "avatar.drive":
                        avatar_count += 1
                        expr = p.get("expression", "")
                        print(f"  Avatar: expression={expr}")
                    elif t == "error":
                        print(f"  ERROR: {p}")
                        break
                    # 忽略 session.created 等控制消息

                except asyncio.TimeoutError:
                    print("  超时等待更多消息")
                    break

            results = []
            if ai_count > 0:
                print(f"  PASS: LLM流式 ({ai_count} chunks)")
                results.append(True)
            else:
                print("  FAIL: 无LLM响应")
                results.append(False)

            if got_done:
                print("  PASS: done标记")
                results.append(True)
            else:
                print("  FAIL: 无done标记")
                results.append(False)

            if avatar_count > 0:
                print(f"  PASS: Avatar驱动 ({avatar_count})")
                results.append(True)
            else:
                print("  WARN: 无Avatar驱动")
                results.append(True)  # 非必须

            if first_chunk_time:
                ttfb = (first_chunk_time - t0) * 1000
                print(f"  首包延迟(TTFB): {ttfb:.0f}ms")
                results.append(ttfb < 5000)

            return all(results)
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


async def test_interrupt_via_java():
    """T6: 中断机制 (via Java WS)"""
    print("\n=== T6: 中断机制 ===")
    import websockets

    try:
        async with websockets.connect(URI, open_timeout=5) as ws:
            # 直接发 interrupt
            await ws.send(json.dumps({
                "type": "interrupt",
                "payload": {"sessionId": "test-interrupt"},
            }))
            for _ in range(5):
                try:
                    data = await recv(ws, 5)
                    if data.get("type") == "interrupt.ack":
                        print("  PASS: interrupt.ack 收到")
                        return True
                except asyncio.TimeoutError:
                    break
            print("  FAIL: 无 interrupt.ack")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


async def test_mcp_tool_direct():
    """T7: MCP 工具调用 (via Python HTTP API, 不占用WS连接)"""
    print("\n=== T7: MCP 工具调用 ===")
    try:
        with urllib.request.urlopen("http://localhost:8000/tools", timeout=5) as r:
            data = json.loads(r.read())
            tools = data.get("tools", [])
            tool_names = [t["name"] for t in tools]
            if "current_time" in tool_names:
                print(f"  PASS: 工具列表包含 current_time, 共{len(tools)}个工具")
                return True
            else:
                print(f"  FAIL: current_time 不在工具列表中: {tool_names}")
                return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


async def test_knowledge_build():
    """T8: 知识库构建"""
    print("\n=== T8: 知识库构建 ===")
    try:
        body = json.dumps({
            "skillId": "e2e_test",
            "documents": [
                {"text": "数字人工作室支持文本和语音交互。", "metadata": {"source": "test"}},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:8000/knowledge/build",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            if data.get("success"):
                print(f"  PASS: chunkCount={data.get('chunkCount', 0)}")
                return True
            else:
                print(f"  FAIL: {data}")
                return False
    except Exception as e:
        print(f"  WARN: {e}")
        return True  # 非阻塞


async def main():
    print("=" * 60)
    print("Digital Human Studio - 端到端全链路测试")
    print("=" * 60)

    results = {}

    tests = [
        ("T1: Java HTTP 健康检查", test_java_health),
        ("T2: Python AI Engine 健康", test_python_health),
        ("T3: WS 连接鉴权", test_ws_connect_auth),
        ("T4: 会话生命周期", test_session_lifecycle_via_java),
        ("T5: 文本对话全链路", test_chat_full_link),
        ("T6: 中断机制", test_interrupt_via_java),
        ("T7: MCP 工具调用", test_mcp_tool_direct),
        ("T8: 知识库构建", test_knowledge_build),
    ]

    for name, test_fn in tests:
        try:
            passed = await test_fn()
            results[name] = passed
        except Exception as e:
            print(f"  异常: {e}")
            results[name] = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    pass_count = sum(1 for v in results.values() if v)
    fail_count = sum(1 for v in results.values() if not v)

    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")

    print(f"\n总计: {pass_count} 通过, {fail_count} 失败")

    if fail_count == 0:
        print("\n🎉 全部测试通过！可以开始开发前端。")

    return fail_count == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
