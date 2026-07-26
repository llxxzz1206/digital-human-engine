from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.rag.milvus_client import milvus_manager
from app.rag.embedding import embedding_service
from app.rag.document_parser import parse_document, get_chunk_size, DocumentParseError

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """知识库构建器：文档解析 + 自适应切片 + Embedding 写入 Milvus"""

    def __init__(self) -> None:
        self._default_chunk_size = 500
        self._chunk_overlap = 80  # 切片重叠字符数

    async def build(
        self,
        skill_id: str,
        documents: list[dict[str, Any]],
    ) -> int:
        """构建知识库（纯文本输入）

        Args:
            skill_id: Skill ID，用于确定 collection 名
            documents: 文档列表，每个文档包含 {"text": str, "metadata": dict}

        Returns:
            写入的向量数量
        """
        collection_name = f"skill_{skill_id}"
        milvus_manager.ensure_collection(collection_name)

        all_chunks: list[dict[str, Any]] = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            file_type = metadata.get("file_type", "txt")
            chunk_size = get_chunk_size(file_type)
            chunks = self._split_text(text, metadata, chunk_size)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("无切片可写入: skill_id=%s", skill_id)
            return 0

        return await self._embed_and_insert(collection_name, all_chunks, skill_id)

    async def build_from_file(
        self,
        skill_id: str,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """从文件构建知识库

        Args:
            skill_id: Skill ID
            file_path: 上传文件路径（PDF/Word/TXT/MD）

        Returns:
            {"chunks": int, "filename": str, "file_type": str, "char_count": int}

        Raises:
            DocumentParseError: 文件解析失败
        """
        # 解析文档
        parsed = parse_document(file_path)
        text = parsed["text"]
        metadata = parsed["metadata"]

        # 自适应分块
        file_type = metadata.get("file_type", "txt")
        chunk_size = get_chunk_size(file_type)
        chunks = self._split_text(text, metadata, chunk_size)

        if not chunks:
            raise DocumentParseError(f"文档切片结果为空: {metadata.get('filename', 'unknown')}")

        # 写入向量库
        collection_name = f"skill_{skill_id}"
        milvus_manager.ensure_collection(collection_name)
        count = await self._embed_and_insert(collection_name, chunks, skill_id)

        logger.info(
            "文件知识库构建完成: skill=%s, file=%s, chunks=%d",
            skill_id, metadata.get("filename"), count,
        )

        return {
            "chunks": count,
            "filename": metadata.get("filename", ""),
            "file_type": file_type,
            "char_count": len(text),
            "page_count": metadata.get("page_count"),
        }

    async def _embed_and_insert(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
        skill_id: str,
    ) -> int:
        """批量向量化并写入 Milvus"""
        texts = [chunk["text"] for chunk in chunks]
        vectors = await embedding_service.embed_batch(texts)

        # 检查是否有失败（None 表示失败）
        failed_count = sum(1 for v in vectors if v is None)
        if failed_count > 0:
            logger.error(
                "Embedding 部分失败: skill=%s, failed=%d/%d",
                skill_id, failed_count, len(vectors),
            )
            # 过滤掉失败的
            valid_pairs = [
                (chunk, vec) for chunk, vec in zip(chunks, vectors) if vec is not None
            ]
            if not valid_pairs:
                raise RuntimeError("Embedding 全部失败，请检查 API Key 和网络连接")
        else:
            valid_pairs = list(zip(chunks, vectors))

        client = milvus_manager.get_client()
        data = []
        for chunk, vector in valid_pairs:
            data.append({
                "vector": vector,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })

        client.insert(collection_name=collection_name, data=data)
        logger.info("知识库已构建: skill_id=%s, chunks=%d", skill_id, len(data))
        return len(data)

    def _split_text(
        self,
        text: str,
        metadata: dict[str, Any],
        chunk_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """文本切片（按固定长度 + 重叠，优先按段落边界切分）"""
        size = chunk_size or self._default_chunk_size

        if len(text) <= size:
            return [{"text": text, "metadata": metadata}]

        chunks = []
        # 先按段落分割，再合并到目标大小
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append({"text": current_chunk, "metadata": metadata})
                # 段落本身超长，强制按字数切
                if len(para) > size:
                    start = 0
                    while start < len(para):
                        end = start + size
                        chunks.append({"text": para[start:end], "metadata": metadata})
                        start = end - self._chunk_overlap
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append({"text": current_chunk, "metadata": metadata})

        return chunks


knowledge_builder = KnowledgeBuilder()
