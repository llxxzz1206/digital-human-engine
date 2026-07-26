"""验证当前配置"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.settings import settings

print("=" * 60)
print("当前配置验证")
print("=" * 60)

print("\n【LLM 配置】")
print(f"  Provider: {settings.llm.provider}")
print(f"  Model: {settings.llm.model}")
print(f"  Fast Model: {settings.llm.fast_model}")
print(f"  API Base: {settings.llm.api_base}")
print(f"  API Key: {settings.llm.api_key[:20]}..." if settings.llm.api_key else "  API Key: 未配置")

print("\n【RAG 配置】")
print(f"  Rerank Enabled: {settings.rag.rerank_enabled}")
print(f"  Top K: {settings.rag.top_k}")

print("\n【ASR 配置】")
print(f"  Provider: {settings.asr.provider}")
print(f"  Model: {settings.asr.model}")

print("\n【TTS 配置】")
print(f"  Voice: {settings.tts.voice}")
print(f"  Speed: {settings.tts.speed}")

print("\n" + "=" * 60)

# 验证配置是否正确
if settings.llm.provider == "zhipu" and settings.llm.model == "glm-4-flash":
    print("✅ LLM 配置正确：GLM-4-Flash")
else:
    print(f"❌ LLM 配置错误：{settings.llm.provider}/{settings.llm.model}")

if not settings.rag.rerank_enabled:
    print("✅ Rerank 已禁用")
else:
    print("❌ Rerank 仍启用")

print("=" * 60)