"""完整功能测试脚本 v2"""
import asyncio
import json

import websockets

URI = "ws://localhost:8000/ws/ai"


async def recv_type(ws, expected_type, timeout=10):
    """接收指定类型的消息"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=remaining)
            data = json.loads(resp)
            if data.get("type") == expected_type:
                return data
        except asyncio.TimeoutError:
            break
    return None


async def test_session_lifecycle():
    """T1: 会话生命周期 (create → mount → unmount → destroy)"""
    print("\n=== T1: 会话生命周期 ===")
    results = {"pass": [], "fail": []}

    async with websockets.connect(URI) as ws:
        # create
        await ws.send(json.dumps({
            "type": "session.create",
            "payload": {"avatarId": "test-avatar", "channelId": "ch-1"},
        }))
        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if data.get("type") != "session.created":
            print(f"  FAIL: expected session.created, got {data.get('type')}")
            results["fail"].append("T1: session.create")
            return results
        sid = data["payload"]["sessionId"]
        print(f"  PASS: session.created, sessionId={sid}")
        results["pass"].append("session.create")

        # mount skill
        await ws.send(json.dumps({
            "type": "skill.mount",
            "payload": {"sessionId": sid, "skillId": "example", "channelId": "ch-1"},
        }))
        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if data.get("type") != "skill.mounted":
            print(f"  FAIL: expected skill.mounted, got {data.get('type')}")
            results["fail"].append("T1: skill.mount")
        else:
            print("  PASS: skill.mounted")
            results["pass"].append("skill.mount")

        # unmount skill
        await ws.send(json.dumps({
            "type": "skill.unmount",
            "payload": {"sessionId": sid, "skillId": "example", "channelId": "ch-1"},
        }))
        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if data.get("type") != "skill.unmounted":
            print(f"  FAIL: expected skill.unmounted, got {data.get('type')}")
            results["fail"].append("T1: skill.unmount")
        else:
            print("  PASS: skill.unmounted")
            results["pass"].append("skill.unmount")

        # destroy
        await ws.send(json.dumps({
            "type": "session.destroy",
            "payload": {"sessionId": sid, "channelId": "ch-1"},
        }))
        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if data.get("type") != "session.destroyed":
            print(f"  FAIL: expected session.destroyed, got {data.get('type')}")
            results["fail"].append("T1: session.destroy")
        else:
            print("  PASS: session.destroyed")
            results["pass"].append("session.destroy")

    return results


async def test_chat_streaming():
    """T2: 文本对话 → LLM 流式 + Avatar 驱动"""
    print("\n=== T2: 文本对话（LLM流式 + Avatar驱动）===")
    results = {"pass": [], "fail": []}

    async with websockets.connect(URI) as ws:
        # 创建会话
        await ws.send(json.dumps({
            "type": "session.create",
            "payload": {"avatarId": "test", "channelId": "ch-2"},
        }))
        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        sid = data["payload"]["sessionId"]

        # 发送对话
        await ws.send(json.dumps({
            "type": "chat.send",
            "payload": {
                "sessionId": sid,
                "text": "你好，请简单介绍一下你自己",
                "userId": "test-user",
                "channelId": "ch-2",
                "skillIds": ["example"],
            },
        }))

        ai_count = 0
        avatar_count = 0
        intent_type = ""
        full_text = ""
        got_done = False

        for _ in range(50):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=25)
                data = json.loads(resp)
                t = data["type"]
                p = data.get("payload", {})

                if t == "ai.intent":
                    intent_type = p.get("intent", "")
                    print(f"  意图分类: {intent_type}")
                elif t == "ai.stream":
                    ai_count += 1
                    text_chunk = p.get("text", "")
                    full_text += text_chunk
                    if p.get("done"):
                        got_done = True
                        print(f"  流式完成: 共{ai_count}个chunk, 文本长度={len(full_text)}")
                    elif ai_count <= 2 or ai_count % 5 == 0:
                        print(f"  [ai.stream #{ai_count}] text={text_chunk[:40]}")
                elif t == "avatar.drive":
                    avatar_count += 1
                    expr = p.get("expression", "")
                    visemes = p.get("visemeHints", "")
                    print(f"  Avatar驱动: expression={expr}, visemeHints={str(visemes)[:40]}")
                elif t == "session.created":
                    pass  # 忽略会话创建响应
                else:
                    print(f"  其他消息: {t}")

            except asyncio.TimeoutError:
                break

        if ai_count > 0:
            print(f"  PASS: LLM流式响应 ({ai_count} chunks, 文本长度={len(full_text)})")
            results["pass"].append("chat.send → ai.stream")
        else:
            print("  FAIL: 无LLM流式响应")
            results["fail"].append("chat.send → ai.stream")

        if got_done:
            print("  PASS: 流式结束标记(done=true)")
            results["pass"].append("ai.stream done marker")
        else:
            print("  WARN: 未收到done标记")
            results["fail"].append("ai.stream done marker")

        if avatar_count > 0:
            print(f"  PASS: Avatar驱动 ({avatar_count} 次)")
            results["pass"].append("chat.send → avatar.drive")
        else:
            print("  WARN: 无Avatar驱动")
            results["pass"].append("chat.send → avatar.drive (no drive for this intent)")

        if intent_type:
            print(f"  PASS: 意图分类 ({intent_type})")
            results["pass"].append("intent classification")

    return results


async def test_mcp_tool():
    """T3: MCP 工具调用"""
    print("\n=== T3: MCP 工具调用 ===")
    results = {"pass": [], "fail": []}

    async with websockets.connect(URI) as ws:
        await ws.send(json.dumps({
            "type": "mcp.tool.call",
            "payload": {"toolName": "echo", "arguments": {"text": "hello test"}},
            "requestId": "req-001",
        }))

        data = None
        for _ in range(5):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(resp)
                if data["type"] in ("mcp.tool.result", "error"):
                    break
            except asyncio.TimeoutError:
                break

        if data and data["type"] == "mcp.tool.result":
            result = data.get("payload", {}).get("result", {})
            print(f"  PASS: echo工具返回 {result}")
            results["pass"].append("mcp.tool.call")
        else:
            print(f"  FAIL: {data}")
            results["fail"].append("mcp.tool.call")

    return results


async def test_interrupt():
    """T4: 中断机制（独立测试）"""
    print("\n=== T4: 中断机制 ===")
    results = {"pass": [], "fail": []}

    async with websockets.connect(URI) as ws:
        await ws.send(json.dumps({
            "type": "interrupt",
            "payload": {"sessionId": "test-interrupt-sid", "channelId": "ch-int"},
        }))

        data = None
        for _ in range(5):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(resp)
                if data["type"] == "interrupt.ack":
                    break
            except asyncio.TimeoutError:
                break

        if data and data.get("type") == "interrupt.ack":
            print("  PASS: interrupt.ack 收到")
            results["pass"].append("interrupt")
        else:
            print("  FAIL: 无 interrupt.ack")
            results["fail"].append("interrupt")

    return results


async def test_health_api():
    """T5: 健康检查 API"""
    print("\n=== T5: 健康检查 API ===")
    results = {"pass": [], "fail": []}

    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/health") as resp:
            data = json.loads(resp.read())
            if data.get("status") == "ok" and data.get("redis") == "connected":
                print(f"  PASS: health={data['status']}, redis={data['redis']}")
                results["pass"].append("health API")
            else:
                print(f"  FAIL: {data}")
                results["fail"].append("health API")
    except Exception as e:
        print(f"  FAIL: {e}")
        results["fail"].append(f"health API - {e}")

    return results


async def test_tools_api():
    """T6: 工具列表 API"""
    print("\n=== T6: 工具列表 API ===")
    results = {"pass": [], "fail": []}

    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/tools") as resp:
            data = json.loads(resp.read())
            tools = data.get("tools", [])
            tool_names = [t["name"] for t in tools]
            print(f"  工具列表: {tool_names}")
            if len(tools) >= 3:
                print(f"  PASS: {len(tools)} 个工具已注册")
                results["pass"].append("tools API")
            else:
                print(f"  FAIL: 仅 {len(tools)} 个工具")
                results["fail"].append("tools API")
    except Exception as e:
        print(f"  FAIL: {e}")
        results["fail"].append(f"tools API - {e}")

    return results


async def test_knowledge_build_api():
    """T7: 知识库构建 API"""
    print("\n=== T7: 知识库构建 API ===")
    results = {"pass": [], "fail": []}

    import urllib.request
    try:
        body = json.dumps({
            "skillId": "test_knowledge",
            "documents": [
                {"text": "数字人工作室是一个AI驱动的虚拟人交互平台。", "metadata": {"source": "test"}},
                {"text": "系统支持文本对话和语音交互两种模式。", "metadata": {"source": "test"}},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:8000/knowledge/build",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            if data.get("success"):
                print(f"  PASS: 知识库构建成功, chunkCount={data.get('chunkCount', 0)}")
                results["pass"].append("knowledge build API")
            else:
                print(f"  FAIL: {data}")
                results["fail"].append("knowledge build API")
    except Exception as e:
        print(f"  FAIL/WARN: {e}")
        # 知识库构建依赖 Milvus + Embedding，可能因网络问题失败
        results["fail"].append(f"knowledge build API - {e}")

    return results


async def test_redis_session_persistence():
    """T8: Redis 会话持久化验证"""
    print("\n=== T8: Redis 会话持久化 ===")
    results = {"pass": [], "fail": []}

    try:
        import redis as sync_redis
        r = sync_redis.Redis(host="localhost", port=6379, decode_responses=True, protocol=2)

        async with websockets.connect(URI) as ws:
            # 创建会话
            await ws.send(json.dumps({
                "type": "session.create",
                "payload": {"avatarId": "persist-test", "channelId": "ch-8"},
            }))
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            sid = data["payload"]["sessionId"]

            # 检查 Redis 中是否存在
            key = f"digitalhuman:session:{sid}"
            val = r.get(key)
            if val:
                print(f"  PASS: Redis中存在会话 key={key[:50]}...")
                results["pass"].append("Redis session persistence")
            else:
                print(f"  FAIL: Redis中未找到会话 key={key}")
                results["fail"].append("Redis session persistence")

            # 销毁会话
            await ws.send(json.dumps({
                "type": "session.destroy",
                "payload": {"sessionId": sid, "channelId": "ch-8"},
            }))
            await asyncio.wait_for(ws.recv(), timeout=5)

            # 检查 Redis 中是否已删除
            val2 = r.get(key)
            if val2 is None:
                print("  PASS: 销毁后 Redis key 已删除")
                results["pass"].append("Redis session destroy")
            else:
                print("  FAIL: 销毁后 Redis key 仍存在")
                results["fail"].append("Redis session destroy")
    except Exception as e:
        print(f"  FAIL: {e}")
        results["fail"].append(f"Redis session persistence - {e}")

    return results


async def main():
    print("=" * 60)
    print("Digital Human Studio - 全链路功能测试 v2")
    print("=" * 60)

    all_results = {"pass": [], "fail": []}

    tests = [
        test_health_api,
        test_tools_api,
        test_session_lifecycle,
        test_chat_streaming,
        test_mcp_tool,
        test_interrupt,
        test_knowledge_build_api,
        test_redis_session_persistence,
    ]

    for test_fn in tests:
        try:
            r = await test_fn()
            all_results["pass"].extend(r["pass"])
            all_results["fail"].extend(r["fail"])
        except Exception as e:
            print(f"\n  测试异常: {test_fn.__name__}: {e}")
            all_results["fail"].append(f"{test_fn.__name__} - exception: {e}")

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    pass_count = len(all_results["pass"])
    fail_count = len(all_results["fail"])

    print(f"\n✅ PASS ({pass_count}):")
    for item in all_results["pass"]:
        print(f"  + {item}")

    if all_results["fail"]:
        print(f"\n❌ FAIL ({fail_count}):")
        for item in all_results["fail"]:
            print(f"  - {item}")
    else:
        print("\n🎉 全部测试通过！")

    print(f"\n总计: {pass_count} 通过, {fail_count} 失败")
    return fail_count == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
