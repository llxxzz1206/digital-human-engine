"""测试 GLM-4-Flash 连接"""
import asyncio
from openai import AsyncOpenAI

API_KEY = "7bcb706aeebe4306a4ed993cec709e1e.GRdNP5AIyEJmrt7s"
API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"


async def test_connection():
    """测试连接"""
    print("=" * 60)
    print("GLM-4-Flash 连接测试")
    print("=" * 60)

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)

    # 测试 1：简单问答
    print("\n测试 1：简单问答")
    print("-" * 40)
    import time
    start = time.time()

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=50,
    )

    elapsed = (time.time() - start) * 1000
    reply = response.choices[0].message.content
    print(f"回复: {reply}")
    print(f"耗时: {elapsed:.0f}ms")

    # 测试 2：流式输出
    print("\n测试 2：流式输出（测试首字延迟）")
    print("-" * 40)

    start = time.time()
    first_chunk_time = None

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "请用一句话介绍博物馆"}],
        max_tokens=50,
        stream=True,
    )

    full_reply = ""
    chunk_count = 0
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttft = (first_chunk_time - start) * 1000
                print(f"首字延迟（TTFT）: {ttft:.0f}ms")
            content = chunk.choices[0].delta.content
            full_reply += content
            chunk_count += 1

    elapsed = (time.time() - start) * 1000
    print(f"完整回复: {full_reply}")
    print(f"总耗时: {elapsed:.0f}ms")
    print(f"分片数: {chunk_count}")

    print("\n" + "=" * 60)
    print("✅ 连接测试成功！")
    print("=" * 60)

    # 性能总结
    print("\n性能总结：")
    print(f"- 首字延迟（TTFT）: {ttft:.0f}ms")
    print(f"- 总耗时: {elapsed:.0f}ms")
    print(f"- 模型: {MODEL}")
    print(f"- API Base: {API_BASE}")

    if ttft < 500:
        print("\n🎉 性能优秀！首字延迟 < 500ms")
    elif ttft < 1000:
        print("\n✅ 性能良好！首字延迟 < 1000ms")
    else:
        print("\n⚠️ 性能一般，建议检查网络连接")


if __name__ == "__main__":
    asyncio.run(test_connection())