from __future__ import annotations

import json
import logging

from app.rag.knowledge_builder import knowledge_builder

logger = logging.getLogger(__name__)


class KnowledgeBuildConsumer:
    """RocketMQ 知识库构建消费者

    Java 后端投递知识库构建消息到 RocketMQ，
    Python 端消费消息后调用 KnowledgeBuilder 构建知识库。

    消息格式（JSON）：
    {
        "skillId": "example",
        "documents": [
            {"text": "文档内容...", "metadata": {"source": "file.pdf", "page": 1}},
            ...
        ]
    }
    """

    def __init__(self) -> None:
        self._consumer = None
        self._running = False

    async def start(self) -> None:
        """启动 RocketMQ 消费者"""
        try:
            from rocketmq.client import PushConsumer, ConsumeStatus
            from app.config.settings import settings

            namesrv = f"{settings.redis.host}:9876"  # RocketMQ NameServer 地址
            # 使用独立的 RocketMQ 配置
            namesrv_addr = "localhost:9876"

            self._consumer = PushConsumer("knowledge-build-consumer")
            self._consumer.set_name_server_address(namesrv_addr)
            self._consumer.subscribe("knowledge-build-topic", self._on_message)
            self._consumer.start()
            self._running = True
            logger.info("RocketMQ 知识库构建消费者已启动")
        except ImportError:
            logger.warning("rocketmq-client-python 未安装，异步知识库构建不可用")
        except Exception as e:
            logger.error("RocketMQ 消费者启动失败: %s", e)

    def _on_message(self, msg) -> None:
        """消息回调"""
        try:
            body = json.loads(msg.body.decode("utf-8"))
            skill_id = body.get("skillId", "")
            documents = body.get("documents", [])

            logger.info("收到知识库构建消息: skillId=%s, docs=%d", skill_id, len(documents))

            # 同步执行构建（在独立线程中）
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                count = loop.run_until_complete(
                    knowledge_builder.build(skill_id, documents)
                )
                logger.info("知识库构建完成: skillId=%s, chunks=%d", skill_id, count)
            finally:
                loop.close()

        except Exception as e:
            logger.error("知识库构建消息处理失败: %s", e)

        return 0  # CONSUME_SUCCESS

    async def stop(self) -> None:
        """停止消费者"""
        if self._consumer and self._running:
            self._consumer.shutdown()
            self._running = False
            logger.info("RocketMQ 知识库构建消费者已停止")


knowledge_build_consumer = KnowledgeBuildConsumer()
