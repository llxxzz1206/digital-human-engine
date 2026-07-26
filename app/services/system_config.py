"""系统配置服务 - 集中管理 LLM/ASR/TTS/RAG 等配置"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.infrastructure.redis import RedisPool

logger = logging.getLogger(__name__)

# Redis Key
_CONFIG_PREFIX = "digitalhuman:config:"

# 默认配置
DEFAULT_CONFIG = {
    "llm": {
        "provider": "zhipu",
        "model": "glm-4-flash",
        "fast_model": "",
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "temperature": 0.7,
        "max_tokens": 150,
    },
    "asr": {
        "provider": "whisper",
        "model": "small",
        "hotwords": "",
    },
    "tts": {
        "voice": "x4_lingxiaoxuan_oral",
        "speed": 50,
        "volume": 50,
    },
    "rag": {
        "rerank_enabled": False,
        "top_k": 3,
        "threshold_a": 0.85,
        "threshold_b": 0.5,
    },
}


class SystemConfigService:
    """系统配置服务

    功能：
    1. 集中管理所有系统配置（LLM/ASR/TTS/RAG）
    2. 支持热更新（无需重启服务）
    3. 配置持久化（Redis 存储）
    4. 配置验证
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._fallback: dict[str, Any] = {}

    async def _get_redis(self):
        """获取 Redis 客户端"""
        try:
            redis = await RedisPool.get()
            if await redis.ping():
                return redis
        except Exception:
            pass
        return None

    async def get_config(self, category: str = "") -> dict[str, Any]:
        """获取配置

        Args:
            category: 配置类型（llm/asr/tts/rag），空字符串返回全部

        Returns:
            配置字典
        """
        redis = await self._get_redis()

        if redis:
            # 从 Redis 读取
            if category:
                key = f"{_CONFIG_PREFIX}{category}"
                data = await redis.get(key)
                if data:
                    return json.loads(data)
                # 返回默认值
                return DEFAULT_CONFIG.get(category, {})
            else:
                # 读取全部配置
                result = {}
                for cat in ["llm", "asr", "tts", "rag"]:
                    key = f"{_CONFIG_PREFIX}{cat}"
                    data = await redis.get(key)
                    result[cat] = json.loads(data) if data else DEFAULT_CONFIG.get(cat, {})
                return result
        else:
            # 内存回退
            if category:
                return self._fallback.get(category, DEFAULT_CONFIG.get(category, {}))
            return {cat: self._fallback.get(cat, DEFAULT_CONFIG.get(cat, {})) for cat in ["llm", "asr", "tts", "rag"]}

    async def update_config(
        self,
        category: str,
        config: dict[str, Any],
        validate: bool = True,
    ) -> tuple[bool, str]:
        """更新配置

        Args:
            category: 配置类型（llm/asr/tts/rag）
            config: 新配置
            validate: 是否验证配置

        Returns:
            (success, message)
        """
        if category not in ["llm", "asr", "tts", "rag"]:
            return False, f"无效的配置类型: {category}"

        # 验证配置
        if validate:
            valid, msg = await self._validate_config(category, config)
            if not valid:
                return False, msg

        # 保存配置
        redis = await self._get_redis()
        key = f"{_CONFIG_PREFIX}{category}"

        if redis:
            await redis.set(key, json.dumps(config, ensure_ascii=False))
            logger.info("配置已更新: %s", category)
        else:
            self._fallback[category] = config

        # 记录变更历史
        await self._record_change(category, config)

        return True, "配置更新成功"

    async def reset_config(self, category: str) -> bool:
        """重置配置为默认值"""
        if category not in DEFAULT_CONFIG:
            return False

        default = DEFAULT_CONFIG[category].copy()
        success, _ = await self.update_config(category, default, validate=False)
        return success

    async def _validate_config(self, category: str, config: dict[str, Any]) -> tuple[bool, str]:
        """验证配置有效性"""
        if category == "llm":
            # 验证 LLM 配置
            provider = config.get("provider", "")
            if provider not in ["qwen", "zhipu", "deepseek", "openai", "mock"]:
                return False, f"不支持的 LLM 提供者: {provider}"

            # 验证 API Key（非 mock 模式）
            # 实际验证需要调用 API，这里只做格式检查

        elif category == "asr":
            provider = config.get("provider", "")
            if provider not in ["whisper", "xunfei", "disabled"]:
                return False, f"不支持的 ASR 提供者: {provider}"

        elif category == "tts":
            speed = config.get("speed", 50)
            if not (0 <= speed <= 100):
                return False, "语速必须在 0-100 之间"

        elif category == "rag":
            top_k = config.get("top_k", 3)
            if not (1 <= top_k <= 10):
                return False, "top_k 必须在 1-10 之间"

        return True, "验证通过"

    async def _record_change(self, category: str, config: dict[str, Any]) -> None:
        """记录配置变更历史"""
        redis = await self._get_redis()
        if not redis:
            return

        # 记录到历史列表（最近 100 条）
        history_key = f"{_CONFIG_PREFIX}history:{category}"
        record = {
            "timestamp": int(time.time() * 1000),
            "config": config,
        }
        await redis.lpush(history_key, json.dumps(record, ensure_ascii=False))
        await redis.ltrim(history_key, 0, 99)  # 只保留最近 100 条

    async def get_change_history(self, category: str, limit: int = 10) -> list[dict]:
        """获取配置变更历史"""
        redis = await self._get_redis()
        if not redis:
            return []

        history_key = f"{_CONFIG_PREFIX}history:{category}"
        records = await redis.lrange(history_key, 0, limit - 1)
        return [json.loads(r) for r in records]

    async def test_llm_connection(self, provider: str, model: str, api_key: str, api_base: str) -> tuple[bool, str]:
        """测试 LLM 连接"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
            )
            return True, "连接成功"
        except Exception as e:
            return False, f"连接失败: {str(e)}"


# 全局实例
system_config = SystemConfigService()