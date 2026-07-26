"""P0 部署改造验证脚本

用法:
  1. 启动后端: uvicorn app.main:app --port 8000
  2. 设置环境变量（或直接在脚本里改）:
     ADMIN_TOKEN=你的token
  3. 运行: python scripts/verify_p0.py

验证项:
  [1] Admin API 无 token → 401
  [2] Admin API 带 token → 200
  [3] WS 频率限制 → 第 11 次连接被拒 (4002)
  [4] Avatar 动作配置化 → 未知手势降级 talking
"""
import asyncio
import os
import sys

import httpx

BASE_URL = os.environ.get("DH_BASE_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "test-token-123")
WS_URL = BASE_URL.replace("http", "ws") + "/ws"

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


async def test_admin_auth():
    """[1][2] Admin API 鉴权"""
    print("\n== Admin API 鉴权 ==")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as client:
        # 无 token → 401
        r = await client.get("/admin/skill/list")
        report("无 token 返回 401", r.status_code == 401, f"实际: {r.status_code}")

        # 错误 token → 401
        r = await client.get("/admin/skill/list", headers={"X-Admin-Token": "wrong"})
        report("错误 token 返回 401", r.status_code == 401, f"实际: {r.status_code}")

        # 正确 token → 200
        r = await client.get("/admin/skill/list", headers={"X-Admin-Token": ADMIN_TOKEN})
        report("正确 token 返回 200", r.status_code == 200, f"实际: {r.status_code}")


async def test_ws_rate_limit():
    """[3] WS 频率限制"""
    print("\n== WebSocket 频率限制 ==")
    try:
        import websockets
    except ImportError:
        report("websockets 库未安装", False, "pip install websockets")
        return

    results = []
    for i in range(12):
        try:
            async with websockets.connect(WS_URL, close_timeout=2) as ws:
                # 连接成功，立即关闭
                results.append("connected")
        except Exception as e:
            err_str = str(e)
            if "4002" in err_str or "Rate limit" in err_str or "403" in err_str:
                results.append("rate_limited")
            elif "4001" in err_str:
                results.append("auth_rejected")
            else:
                results.append(f"error:{err_str[:50]}")

    # 前 10 次应该连上，第 11/12 次应该被限流
    connected_count = results[:10].count("connected")
    limited = any("rate_limited" in r for r in results[10:])

    report(f"前 10 次连接成功 ({connected_count}/10)", connected_count >= 9,
           f"结果: {results[:10]}")
    report("第 11+ 次被限流 (4002)", limited,
           f"结果: {results[10:]}")


async def test_avatar_gesture_config():
    """[4] Avatar 动作配置化"""
    print("\n== Avatar 动作配置化 ==")

    # 直接测试 AvatarDriver 逻辑（不需要完整 WS 链路）
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.avatar.driver import AvatarDriver, DEFAULT_GESTURES

    # 默认 driver：wave 不在默认列表 → 降级 talking
    driver = AvatarDriver()
    result = await driver.generate_drive("wave")
    report("默认 driver: wave 降级 talking",
           result["state"] == "talking" and result["loop"] is True,
           f"实际: {result}")

    # 配置了 wave 的 driver
    config = {"gestures": "idle,talking,greeting,point_left,point_right,bow,wave", "loopStates": "idle,talking"}
    driver2 = AvatarDriver.from_config(config)
    result2 = await driver2.generate_drive("wave")
    report("配置 wave 后: wave 正常返回",
           result2["state"] == "wave" and result2["loop"] is False,
           f"实际: {result2}")

    # 未知手势降级
    result3 = await driver2.generate_drive("fly")
    report("未配置 fly: 降级 talking",
           result3["state"] == "talking",
           f"实际: {result3}")

    # 空配置用默认值
    driver4 = AvatarDriver.from_config({})
    result4 = await driver4.generate_drive("bow")
    report("空配置: bow 正常（默认六态）",
           result4["state"] == "bow" and result4["loop"] is False,
           f"实际: {result4}")


async def main():
    print("=" * 50)
    print("P0 部署改造验证")
    print(f"目标: {BASE_URL}")
    print("=" * 50)

    # [4] 不需要服务运行，纯逻辑测试
    await test_avatar_gesture_config()

    # [1][2][3] 需要服务运行
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=3) as c:
            await c.get("/health")
    except Exception:
        print(f"\n[SKIP] 服务未启动 ({BASE_URL})，跳过 Admin/WS 测试")
        print("  启动方式: cd python-ai-engine && uvicorn app.main:app --port 8000")
        print(f"\n{'=' * 50}")
        print(f"结果: {passed} passed, {failed} failed (部分跳过)")
        return

    await test_admin_auth()
    await test_ws_rate_limit()

    print(f"\n{'=' * 50}")
    print(f"结果: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
