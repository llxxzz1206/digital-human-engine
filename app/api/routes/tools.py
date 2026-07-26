from __future__ import annotations

from fastapi import APIRouter

from app.mcp.registry import tool_registry
from app.rag.knowledge_builder import knowledge_builder

router = APIRouter()


@router.get("/tools")
async def list_tools() -> dict:
    tools = tool_registry.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "permissions": t.permissions,
            }
            for t in tools
        ],
    }


@router.post("/knowledge/build")
async def build_knowledge(body: dict) -> dict:
    """直接触发知识库构建（开发阶段替代 RocketMQ）

    Body:
    {
        "skillId": "example",
        "documents": [{"text": "文档内容...", "metadata": {"source": "file.pdf"}}]
    }
    """
    skill_id = body.get("skillId", "")
    documents = body.get("documents", [])

    if not skill_id or not documents:
        return {"success": False, "message": "skillId 和 documents 不能为空"}

    count = await knowledge_builder.build(skill_id, documents)
    return {"success": True, "skillId": skill_id, "chunkCount": count}
