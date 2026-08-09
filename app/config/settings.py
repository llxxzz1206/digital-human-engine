from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import Field


class LLMConfig(BaseSettings):
    provider: str = Field(default="mock", description="LLM 提供者 (mock/qwen/openai/deepseek)")
    model: str = Field(default="mock-model", description="模型名称")
    fast_model: str = Field(default="", description="难度路由快模型：rag_chat 等简短落地问答用它降低首字延迟；空=不路由，统一用 model")
    api_key: str = Field(default="", description="API Key")
    api_base: str = Field(default="", description="API Base URL (OpenAI 兼容格式)")
    fallback_providers: str = Field(
        default="",
        description='备用 LLM 列表 JSON，如 [{"model":"qwen-turbo","api_key":"...","api_base":"..."}]，主 provider 失败时依次尝试'
    )


class ASRConfig(BaseSettings):
    provider: str = Field(default="xunfei", description="ASR 引擎 (whisper/xunfei/disabled)")
    model: str = Field(default="small", description="Whisper 模型 (tiny/base/small/medium/large)")
    wake_model: str = Field(default="tiny", description="唤醒词检测专用 Whisper 模型（追求低延迟，默认 tiny）")
    device: str = Field(default="cpu", description="推理设备 (cpu/cuda)")
    # 讯飞 ASR 配置（provider=xunfei 时使用）
    app_id: str = Field(default="", description="讯飞 ASR APP_ID")
    api_key: str = Field(default="", description="讯飞 ASR API_KEY")
    api_secret: str = Field(default="", description="讯飞 ASR API_SECRET")
    eos: int = Field(default=6000, description="讯飞 ASR 静音多少毫秒停止识别（前端已 VAD 截断，此值兜底）")
    hotwords: str = Field(default="", description="讯飞 ASR 会话级热词，竖线分隔（如 挂号|门诊|CT），提升专有词识别率")


class TTSConfig(BaseSettings):
    app_id: str = Field(default="", description="讯飞 TTS APP_ID")
    api_key: str = Field(default="", description="讯飞 TTS API_KEY")
    api_secret: str = Field(default="", description="讯飞 TTS API_SECRET")
    voice: str = Field(default="x4_lingxiaoxuan_oral", description="讯飞发音人")
    speed: int = Field(default=50, description="语速 (0-100)")
    api_url: str = Field(default="wss://tts-api.xfyun.cn/v2/tts", description="讯飞 TTS WebSocket URL")
    api_host: str = Field(default="ws-api.xfyun.cn", description="讯飞 TTS 鉴权 host")


class RedisConfig(BaseSettings):
    host: str = Field(default="localhost", description="Redis 主机")
    port: int = Field(default=6379, description="Redis 端口")
    db: int = Field(default=0, description="Redis 数据库编号")
    password: str = Field(default="", description="Redis 密码")
    pool_size: int = Field(default=10, description="连接池大小")
    session_ttl: int = Field(default=3600, description="会话 TTL（秒），与 Java 端 sessionExpireSeconds 对齐")


class MilvusConfig(BaseSettings):
    host: str = Field(default="localhost", description="Milvus 主机")
    port: int = Field(default=19530, description="Milvus 端口")
    embedding_provider: str = Field(default="openai_compatible", description="Embedding 提供者")
    embedding_model: str = Field(default="qwen3.7-text-embedding", description="Embedding 模型名")
    embedding_dim: int = Field(default=1024, description="向量维度")


class RAGConfig(BaseSettings):
    rerank_enabled: bool = Field(default=False, description="是否启用 Rerank（展馆场景建议关闭，减少延迟）")
    rerank_model: str = Field(default="qwen3-rerank", description="DashScope Rerank 模型名")
    rerank_threshold_a: float = Field(default=0.70, description="FAQ direct 路由阈值：>=a 且来源为 FAQ 时直接输出")
    rerank_threshold_b: float = Field(default=0.4, description="低相关阈值：>=b 注入LLM上下文，<b 普通对话")
    rerank_noise_threshold: float = Field(default=0.10, description="噪音判定阈值：rerank 最高分低于此值 → 判定为噪音/无关输入，不送 LLM（实测噪音约 0.05-0.15，需用真实数据标定）")
    audio_rms_threshold: float = Field(default=0.008, description="L1 VAD 门控：音频 RMS 低于此值视为静音/底噪，不送 ASR（需实测标定）")
    faq_enabled: bool = Field(default=True, description="是否启用 FAQ 双集合架构")
    faq_direct_threshold: float = Field(default=0.70, description="FAQ direct 路由 rerank 分数阈值")
    faq_promotion_threshold: int = Field(default=5, description="FAQ 进入待审核的命中次数阈值（之后还需人工确认）")
    faq_similarity_threshold: float = Field(default=0.92, description="FAQ 相似问题匹配阈值（Milvus cosine）")
    context_window: int = Field(default=40, description="LLM 上下文窗口大小（消息条数，40=20轮对话）")
    top_k: int = Field(default=3, description="RAG 检索返回文档数（展馆场景建议 3）")


class AudioEnhanceConfig(BaseSettings):
    """音频增强配置（展馆嘈杂环境优化）"""
    enabled: bool = Field(default=False, description="是否启用音频增强")
    vad_aggressiveness: int = Field(default=2, ge=0, le=3, description="VAD 灵敏度 (0-3, 0最激进)")
    noise_reduce_strength: float = Field(default=0.7, ge=0.0, le=1.0, description="降噪强度 (0-1)")
    min_speech_duration_ms: int = Field(default=200, description="最小语音时长 (ms)")


class DatabaseConfig(BaseSettings):
    url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/digital_human",
        description="PostgreSQL 连接字符串",
    )


class SandboxConfig(BaseSettings):
    default_timeout: float = Field(default=30.0, description="工具默认超时（秒）")
    max_timeout: float = Field(default=120.0, description="工具最大超时（秒）")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMConfig = LLMConfig()
    asr: ASRConfig = ASRConfig()
    tts: TTSConfig = TTSConfig()
    redis: RedisConfig = RedisConfig()
    milvus: MilvusConfig = MilvusConfig()
    rag: RAGConfig = RAGConfig()
    audio_enhance: AudioEnhanceConfig = AudioEnhanceConfig()
    database: DatabaseConfig = DatabaseConfig()
    sandbox: SandboxConfig = SandboxConfig()
    log_level: str = Field(default="INFO", description="日志级别")
    
    # 安全配置
    websocket_token: str = Field(
        default="",
        description="WebSocket 连接令牌（为空则跳过验证，生产环境建议设置）"
    )
    admin_token: str = Field(
        default="",
        description="Admin API 认证令牌（为空则跳过验证并 warn，生产环境必须设置）"
    )
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="CORS 允许的源，逗号分隔（生产环境配实际域名）"
    )


settings = Settings()
