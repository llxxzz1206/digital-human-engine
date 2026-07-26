"""直接验证服务使用的 LLM 配置"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 从实际运行的服务获取配置
from app.config.settings import settings

print("=" * 60)
print("服务运行时 LLM 配置")
print("=" * 60)
print(f"Provider: {settings.llm.provider}")
print(f"Model: {settings.llm.model}")
print(f"API Base: {settings.llm.api_base}")
print(f"API Key: {settings.llm.api_key[:20]}..." if settings.llm.api_key else "API Key: 未配置")
print("=" * 60)

# 验证
if settings.llm.provider == "zhipu" and settings.llm.model == "glm-4-flash":
    print("✅ 服务正在使用 GLM-4-Flash")
else:
    print(f"❌ 服务使用的是: {settings.llm.provider}/{settings.llm.model}")