from __future__ import annotations

import asyncio
import json
import logging
import threading

from app.config.settings import settings
from app.rag.knowledge_builder import knowledge_builder

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    async def start(self) -> None:
        """启动 RocketMQ 消费者"""
        try:
            from rocketmq.client import ConsumeStatus, PushConsumer

            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name="mq-consumer-loop"
            )
            self._loop_thread.start()

            self._consumer = PushConsumer(settings.rocketmq.consumer_group)
            self._consumer.set_name_server_address(settings.rocketmq.namesrv_addr)
            self._consumer.subscribe(settings.rocketmq.topic, self._on_message)
            self._consumer.start()
            self._running = True
            logger.info(
                "RocketMQ 知识库构建消费者已启动: namesrv=%s, topic=%s",
                settings.rocketmq.namesrv_addr,
                settings.rocketmq.topic,
            )
        except ImportError:
            logger.warning("rocketmq-client-python 未安装，异步知识库构建不可用")
        except Exception as e:
            logger.error("RocketMQ 消费者启动失败: %s", e)

    def _on_message(self, msg) -> int:
        """消息回调（同步，在 RocketMQ 消费线程中调用）"""
        try:
            from rocketmq.client import ConsumeStatus
        except ImportError:
            pass

        try:
            body = json.loads(msg.body.decode("utf-8"))
            skill_id = body.get("skillId", "")
            documents = body.get("documents", [])

            logger.info("收到知识库构建消息: skillId=%s, docs=%d", skill_id, len(documents))

            if self._loop and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._build_with_retry(skill_id, documents), self._loop
                )
                count = future.result(timeout=300)
                logger.info("知识库构建完成: skillId=%s, chunks=%d", skill_id, count)
            else:
                logger.error("事件循环未运行，无法处理消息")
                return _consume_retry()

        except Exception as e:
            logger.error("知识库构建消息处理失败: %s", e)
            return _consume_retry()

        return _consume_success()

    async def _build_with_retry(self, skill_id: str, documents: list[dict]) -> int:
        """带重试的知识库构建"""
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await knowledge_builder.build(skill_id, documents)
            except Exception as e:
                last_error = e
                logger.warning(
                    "知识库构建失败 (attempt %d/%d): skillId=%s, error=%s",
                    attempt, _MAX_RETRIES, skill_id, e,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)
        raise last_error  # type: ignore[misc]

    async def stop(self) -> None:
        """停止消费者"""
        if self._consumer and self._running:
            self._consumer.shutdown()
            self._running = False
            logger.info("RocketMQ 知识库构建消费者已停止")
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)


def _consume_success() -> int:
    try:
        from rocketmq.client import ConsumeStatus
        return ConsumeStatus.CONSUME_SUCCESS
    except ImportError:
        return 0


def _consume_retry() -> int:
    try:
        from rocketmq.client import ConsumeStatus
        return ConsumeStatus.RECONSUME_LATER
    except ImportError:
        return 1


knowledge_build_consumer = KnowledgeBuildConsumer()
