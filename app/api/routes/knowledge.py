"""知识库管理 API：文件上传 → 解析 → 向量化 → Milvus"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.rag.knowledge_builder import knowledge_builder
from app.rag.document_parser import DocumentParseError, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# 注意：vite/nginx 代理会剥掉 /api 前缀，后端路由约定不带 /api（否则代理后 404）
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/upload")
async def upload_and_build(
    file: UploadFile = File(...),
    skill_id: str = Form(...),
):
    """上传文档并构建知识库

    支持格式：PDF, DOCX, TXT, MD
    流程：接收文件 → 解析文本 → 自适应分块 → Embedding → 写入 Milvus
    """
    # 验证文件扩展名
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}，支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # 保存到临时文件
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="kb_upload_"
        ) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)

        # 构建知识库
        result = await knowledge_builder.build_from_file(skill_id, tmp_path)
        return {"success": True, **result}

    except DocumentParseError as e:
        logger.warning("文档解析失败: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error("知识库构建失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("知识库构建异常")
        raise HTTPException(status_code=500, detail=f"构建失败: {e}")
    finally:
        # 清理临时文件
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.post("/build-text")
async def build_from_text(
    skill_id: str = Form(...),
    text: str = Form(...),
    filename: str = Form(default="manual_input.txt"),
):
    """从纯文本构建知识库（手动输入或 API 调用）"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本内容为空")

    try:
        documents = [{
            "text": text,
            "metadata": {"filename": filename, "file_type": "txt"},
        }]
        count = await knowledge_builder.build(skill_id, documents)
        return {"success": True, "chunks": count, "filename": filename}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("文本知识库构建异常")
        raise HTTPException(status_code=500, detail=f"构建失败: {e}")


@router.get("/supported-formats")
async def supported_formats():
    """返回支持的文件格式列表"""
    return {
        "formats": sorted(SUPPORTED_EXTENSIONS),
        "description": {
            ".pdf": "PDF 文档（PyMuPDF 解析，推荐分块 800 字符）",
            ".docx": "Word 文档（python-docx 解析，推荐分块 600 字符）",
            ".txt": "纯文本（推荐分块 500 字符）",
            ".md": "Markdown（推荐分块 600 字符）",
        },
    }
