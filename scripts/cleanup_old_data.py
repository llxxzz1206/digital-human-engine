"""清理旧数据脚本 - 删除医院场景和测试数据"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.infrastructure.redis import RedisPool
from app.infrastructure.database import DatabasePool
from app.rag.milvus_client import milvus_manager


async def cleanup_postgresql():
    """清理 PostgreSQL 数据"""
    print("\n=== 清理 PostgreSQL ===")
    pool = await DatabasePool.get()

    # 删除医院场景（物理删除）
    result = await pool.execute(
        "DELETE FROM dh_scene WHERE id = $1", "scene_hospital"
    )
    print(f"删除医院场景: {result}")

    # 检查剩余场景
    rows = await pool.fetch("SELECT id, name, status FROM dh_scene WHERE delete_flag = 'NOT_DELETE'")
    print(f"剩余场景: {[dict(r) for r in rows]}")


async def cleanup_redis():
    """清理 Redis 数据"""
    print("\n=== 清理 Redis ===")
    r = await RedisPool.get()

    # 删除医院 Skill
    deleted = await r.delete("digitalhuman:skill:hospital")
    print(f"删除医院 Skill: {deleted}")

    # 删除测试 Skill
    deleted = await r.delete("digitalhuman:skill:example")
    print(f"删除测试 Skill: {deleted}")

    # 清理所有会话
    session_keys = []
    async for key in r.scan_iter("digitalhuman:session:*"):
        session_keys.append(key)
    if session_keys:
        deleted = await r.delete(*session_keys)
        print(f"删除会话: {deleted} 个")
    else:
        print("无会话需要清理")

    # 清理所有历史记录
    history_keys = []
    async for key in r.scan_iter("digitalhuman:history:*"):
        history_keys.append(key)
    if history_keys:
        deleted = await r.delete(*history_keys)
        print(f"删除历史记录: {deleted} 个")
    else:
        print("无历史记录需要清理")

    # 清理 timings
    deleted = await r.delete("digitalhuman:timings")
    print(f"删除 timings: {deleted}")

    # 检查剩余 Skill
    skill_keys = []
    async for key in r.scan_iter("digitalhuman:skill:*"):
        skill_keys.append(key)
    print(f"剩余 Skill: {skill_keys}")


async def cleanup_milvus():
    """清理 Milvus 数据"""
    print("\n=== 清理 Milvus ===")
    client = milvus_manager.get_client()

    collections = client.list_collections()
    print(f"当前 collections: {collections}")

    # 删除医院相关 collection
    for coll in ["skill_hospital", "faq_hospital"]:
        if coll in collections:
            try:
                client.drop_collection(coll)
                print(f"删除 collection: {coll}")
            except Exception as e:
                print(f"删除 {coll} 失败: {e}")

    # 检查剩余 collections
    collections = client.list_collections()
    print(f"剩余 collections: {collections}")


async def main():
    """主函数"""
    print("开始清理旧数据...")

    try:
        await cleanup_postgresql()
        await cleanup_redis()
        await cleanup_milvus()
        print("\n=== 清理完成 ===")
    except Exception as e:
        print(f"\n清理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())