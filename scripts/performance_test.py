"""
性能基线测试脚本
测试博物馆场景的端到端延迟

使用方法：
1. 启动 Python AI Engine: cd python-ai-engine && uv run python main.py
2. 运行测试: uv run python scripts/performance_test.py

输出：
- 每轮对话的延迟明细（ASR/RAG/LLM/TTS）
- 平均延迟、P95、最大/最小值
- 瓶颈分析
"""
import asyncio
import json
import time
import statistics
from datetime import datetime
from pathlib import Path

import websockets


# 测试配置
WS_URL = "ws://localhost:8000/ws/ai"
TEST_QUERIES = [
    "博物馆几点开门",
    "青铜方鼎在哪里",
    "有停车场吗",
    "恐龙化石在哪个展厅",
    "怎么预约讲解",
    "门票多少钱",
    "有导览服务吗",
    "餐厅在哪里",
    "可以拍照吗",
    "需要预约吗",
]
SCENE_ID = "museum"
DEVICE_ID = "device_001"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "test-reports"


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.records: list[dict] = []
    
    def add(self, query: str, timings: dict, route: str):
        """添加一条记录"""
        self.records.append({
            "query": query,
            "route": route,
            "timings": timings,
            "timestamp": time.time(),
        })
    
    def calculate_stats(self) -> dict:
        """计算统计指标"""
        if not self.records:
            return {}
        
        # 提取各阶段延迟
        e2e_values = [r["timings"].get("e2e", 0) for r in self.records]
        asr_values = [r["timings"].get("asr", 0) for r in self.records if r["timings"].get("asr")]
        rag_values = [r["timings"].get("rag", {}).get("total", 0) for r in self.records]
        llm_values = [r["timings"].get("llm", {}).get("total", 0) for r in self.records]
        tts_values = [r["timings"].get("tts", {}).get("total", 0) for r in self.records]
        
        def stats(values: list[float]) -> dict:
            if not values:
                return {"avg": 0, "p95": 0, "min": 0, "max": 0}
            return {
                "avg": round(statistics.mean(values), 1),
                "p95": round(statistics.quantiles(values, n=100)[94] if len(values) >= 2 else values[0], 1),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
            }
        
        # 路由统计
        route_counts = {}
        for r in self.records:
            route = r["route"]
            route_counts[route] = route_counts.get(route, 0) + 1
        
        # 瓶颈分析
        avg_times = {
            "ASR": stats(asr_values)["avg"],
            "RAG": stats(rag_values)["avg"],
            "LLM": stats(llm_values)["avg"],
            "TTS": stats(tts_values)["avg"],
        }
        bottleneck = max(avg_times, key=avg_times.get)
        
        return {
            "e2e": stats(e2e_values),
            "asr": stats(asr_values),
            "rag": stats(rag_values),
            "llm": stats(llm_values),
            "tts": stats(tts_values),
            "route_distribution": route_counts,
            "bottleneck": bottleneck,
            "total_queries": len(self.records),
        }
    
    def generate_report(self) -> str:
        """生成测试报告"""
        stats = self.calculate_stats()
        
        report = f"""# 性能基线测试报告

> 测试时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 场景：{SCENE_ID}
> 设备：{DEVICE_ID}

## 测试概览

- 测试轮数：{stats["total_queries"]}
- 端到端平均延迟：{stats["e2e"]["avg"]}ms
- 端到端 P95 延迟：{stats["e2e"]["p95"]}ms

## 各阶段延迟统计

| 阶段 | 平均延迟（ms）| P95（ms）| 最小（ms）| 最大（ms）|
|------|-------------|---------|----------|----------|
| ASR | {stats["asr"]["avg"]} | {stats["asr"]["p95"]} | {stats["asr"]["min"]} | {stats["asr"]["max"]} |
| RAG | {stats["rag"]["avg"]} | {stats["rag"]["p95"]} | {stats["rag"]["min"]} | {stats["rag"]["max"]} |
| LLM | {stats["llm"]["avg"]} | {stats["llm"]["p95"]} | {stats["llm"]["min"]} | {stats["llm"]["max"]} |
| TTS | {stats["tts"]["avg"]} | {stats["tts"]["p95"]} | {stats["tts"]["min"]} | {stats["tts"]["max"]} |
| **端到端** | **{stats["e2e"]["avg"]}** | **{stats["e2e"]["p95"]}** | **{stats["e2e"]["min"]}** | **{stats["e2e"]["max"]}** |

## 路由分布

| 路由 | 数量 | 占比 |
|------|------|------|
"""
        # 路由分布
        total = stats["total_queries"]
        for route, count in stats["route_distribution"].items():
            pct = round(count / total * 100, 1)
            report += f"| {route} | {count} | {pct}% |\n"
        
        report += f"""
## 瓶颈分析

当前瓶颈：**{stats["bottleneck"]}**（平均 {stats[stats["bottleneck"].lower()]["avg"]}ms）

优化建议：
- 如果 ASR 是瓶颈：考虑使用云端 ASR（讯飞）替代本地 Whisper
- 如果 RAG 是瓶颈：检查 Milvus 索引配置，考虑启用 Rerank 缓存
- 如果 LLM 是瓶颈：考虑使用 fast_model 路由或缓存常见回复
- 如果 TTS 是瓶颈：启用 TTS 预缓存，检查讯飞 API 响应时间

## 详细记录

"""
        # 详细记录
        for i, r in enumerate(self.records, 1):
            report += f"{i}. **{r['query']}**\n"
            report += f"   - 路由：{r['route']}\n"
            report += f"   - 延迟：ASR {r['timings'].get('asr', 0)}ms, "
            report += f"RAG {r['timings'].get('rag', {}).get('total', 0)}ms, "
            report += f"LLM {r['timings'].get('llm', {}).get('total', 0)}ms, "
            report += f"TTS {r['timings'].get('tts', {}).get('total', 0)}ms\n"
            report += f"   - 端到端：**{r['timings'].get('e2e', 0)}ms**\n\n"
        
        return report


async def test_single_query(ws, query: str, session_id: str, metrics: PerformanceMetrics):
    """测试单轮对话"""
    print(f"\n测试: {query}")
    
    start_time = time.time()
    timings = {}
    
    # 发送消息
    await ws.send(json.dumps({
        "type": "chat.send",
        "payload": {
            "sessionId": session_id,
            "text": query,
        }
    }))
    
    # 接收回复
    full_text = ""
    route = "chat"
    timings_data = {}
    
    while True:
        msg = json.loads(await ws.recv())
        msg_type = msg.get("type")
        
        if msg_type == "chat.reply":
            full_text += msg["payload"].get("text", "")
        elif msg_type == "chat.reply.end":
            route = msg["payload"].get("route", "chat")
            timings_data = msg["payload"].get("timings", {})
            break
    
    # 计算端到端延迟
    e2e = int((time.time() - start_time) * 1000)
    timings_data["e2e"] = e2e
    
    # 记录指标
    metrics.add(query, timings_data, route)
    
    print(f"  回复: {full_text[:50]}...")
    print(f"  路由: {route}")
    print(f"  延迟: {e2e}ms")
    
    return full_text, route, timings_data


async def run_test():
    """运行测试"""
    print("=" * 60)
    print("数字人性能基线测试")
    print("=" * 60)
    
    metrics = PerformanceMetrics()
    
    async with websockets.connect(WS_URL) as ws:
        # 创建会话
        await ws.send(json.dumps({
            "type": "session.create",
            "payload": {
                "sceneId": SCENE_ID,
                "deviceId": DEVICE_ID,
            }
        }))
        
        msg = json.loads(await ws.recv())
        if msg.get("type") != "session.created":
            print("错误：会话创建失败")
            return
        
        session_id = msg["payload"]["sessionId"]
        print(f"会话创建成功: {session_id}")
        
        # 测试对话
        for query in TEST_QUERIES:
            await test_single_query(ws, query, session_id, metrics)
            await asyncio.sleep(1)  # 间隔 1 秒
    
    # 生成报告
    print("\n" + "=" * 60)
    print("生成测试报告...")
    
    report = metrics.generate_report()
    
    # 保存报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"performance-baseline-{datetime.now().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    
    print(f"报告已保存: {report_path}")
    
    # 打印摘要
    stats = metrics.calculate_stats()
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)
    print(f"总轮数: {stats['total_queries']}")
    print(f"端到端平均延迟: {stats['e2e']['avg']}ms")
    print(f"端到端 P95 延迟: {stats['e2e']['p95']}ms")
    print(f"瓶颈阶段: {stats['bottleneck']}")


def main():
    """主函数"""
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n测试已取消")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()