from __future__ import annotations

import logging

from pymilvus import MilvusClient, DataType

from app.config.settings import settings

logger = logging.getLogger(__name__)


class MilvusManager:
    """Milvus 连接管理器"""

    def __init__(self) -> None:
        self._client: MilvusClient | None = None

    def get_client(self) -> MilvusClient:
        """获取 Milvus 客户端"""
        if self._client is None:
            uri = f"http://{settings.milvus.host}:{settings.milvus.port}"
            self._client = MilvusClient(uri=uri)
            logger.info("Milvus 客户端已连接: %s", uri)
        return self._client

    def ensure_collection(self, collection_name: str, dim: int | None = None) -> None:
        """确保 collection 存在，不存在则创建"""
        dim = dim or settings.milvus.embedding_dim
        client = self.get_client()

        if client.has_collection(collection_name):
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.JSON)

        index_params = client.prepare_index_params()
        index_params.add_index("vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Milvus collection 已创建: %s (dim=%d)", collection_name, dim)

    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Milvus 客户端已关闭")


milvus_manager = MilvusManager()
