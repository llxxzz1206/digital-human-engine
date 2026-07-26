# Digital Human AI Engine

数字人对话后端引擎 —— 基于 FastAPI 的实时语音交互服务，为数字人展馆导览提供 ASR → RAG → LLM → TTS → 动作驱动全链路能力。

> **注意**：本项目仅包含后端服务。完整系统还需配合前端客户端（大屏 Web / Android 一体机 / 手机端）使用。前端通过 WebSocket 连接本服务，协议详见下方「通信协议」章节。

## 功能特性

- **实时语音对话**：WebSocket 全双工链路，ASR（讯飞/Whisper）→ RAG 检索增强 → LLM 流式生成 → TTS（讯飞）逐句合成，首字延迟 < 2s
- **数字人驱动**：纯状态机驱动，输出 `avatar.drive` 指令控制前端视频/2D 动画切换，支持动态配置动作列表
- **Skill 插件化**：每个 Skill 独立 Milvus Collection + 工具集 + 系统提示词，管理台热配置，零代码切换场景
- **智能路由**：四级路由（noise / faq_direct / rag_chat / chat），FAQ 高分命中跳过 LLM，降低延迟和 token 成本
- **Function Calling**：LLM 通过工具调用触发导航、设备控制等结构化动作，工具结果驱动数字人手势
- **LLM 多 Provider 熔断**：主 provider 失败自动切备用，连续 5 次失败熔断 60s
- **安全**：Admin API Token 鉴权、WS 连接频率限制、CORS 白名单

## 技术栈

FastAPI (async) · Milvus · PostgreSQL · Redis · 讯飞 ASR/TTS · Whisper · OpenAI-compatible LLM API · WebSocket

## 快速启动

### 环境要求

- Python 3.11+
- Redis 7+
- PostgreSQL 16+
- Milvus 2.4+（可选，不用 RAG 时可不装）
- FFmpeg（Whisper 本地 ASR 需要）

### 安装

```bash
cd python-ai-engine
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`，填入实际配置：

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM__PROVIDER` | LLM 提供者 | `zhipu` |
| `LLM__MODEL` | 模型名 | `glm-4-flash` |
| `LLM__API_KEY` | API Key | `your-key-here` |
| `LLM__API_BASE` | API 地址 | `https://open.bigmodel.cn/api/paas/v4` |
| `LLM__FALLBACK_PROVIDERS` | 备用 LLM（JSON） | `[{"name":"qwen","model":"qwen-turbo","api_key":"...","api_base":"..."}]` |
| `TTS__APP_ID` / `API_KEY` / `API_SECRET` | 讯飞 TTS 凭证 | — |
| `ASR__PROVIDER` | ASR 引擎 | `xunfei` / `whisper` / `disabled` |
| `REDIS__HOST` / `PORT` / `PASSWORD` | Redis 连接 | `localhost` / `6379` |
| `DATABASE__URL` | PostgreSQL 连接串 | `postgresql://postgres:postgres@localhost:5432/digital_human` |
| `MILVUS__HOST` / `PORT` | Milvus 连接 | `localhost` / `19530` |
| `ADMIN_TOKEN` | 管理接口鉴权令牌 | 生产环境必填 |
| `WEBSOCKET_TOKEN` | WS 连接鉴权令牌 | 生产环境必填 |
| `CORS_ORIGINS` | 允许的跨域源 | `https://your-domain.com` |

### 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：`GET http://localhost:8000/health`

### Docker

```bash
docker build -t digital-human-engine .
docker run -p 8000:8000 --env-file .env digital-human-engine
```

## 通信协议

前端通过 WebSocket 连接 `ws://{host}:8000/ws?token={WEBSOCKET_TOKEN}`，所有消息为统一信封格式：

```json
{"type": "消息类型", "payload": {...}}
```

### 客户端 → 服务端

| type | 用途 |
|------|------|
| `session.create` | 创建会话（声明 capabilities、绑定场景/设备） |
| `audio.stream` | 流式音频分片（唤醒阶段） |
| `audio.end` | 音频段结束（触发 ASR） |
| `chat.send` | 文本消息 |
| `session.close` | 关闭会话 |

### 服务端 → 客户端

| type | 用途 |
|------|------|
| `session.created` | 会话创建成功 |
| `asr.result` | 语音识别结果 |
| `chat.reply` | LLM 流式文本回复（增量） |
| `tts.audio` | TTS 音频分片（PCM 16kHz 16bit mono Base64） |
| `avatar.drive` | 数字人动作指令 `{state, loop}` |
| `chat.done` | 回复结束 |

### Capabilities 协商

`session.create` 时声明客户端能力：

- `["text", "audio", "avatar"]`：大屏（文字 + 语音 + 视频数字人）
- `["text", "audio"]`：手机端（文字 + 语音，无数字人视频）
- `["text"]`：纯文本客户端

后端根据 capabilities 决定是否发送 TTS 音频和 avatar.drive 指令。

## 项目结构

```
app/
├── main.py                 # FastAPI 入口 + WS 端点
├── config/settings.py      # 配置（pydantic-settings，读 .env）
├── api/
│   ├── deps.py             # Admin API 鉴权依赖
│   └── routes/             # REST 接口（admin/health/faq/knowledge/voice）
├── ws/
│   └── message_handler.py  # WS 消息分发 + 会话管理
├── workflow/
│   ├── interaction_graph.py # 工作流图（DAG 编排）
│   ├── nodes.py            # 各节点实现（路由/检索/生成/TTS/驱动）
│   └── state.py            # 工作流状态定义
├── llm/service.py          # LLM 服务（流式 + Function Calling + fallback）
├── rag/
│   ├── retriever.py        # RAG 检索（双集合 FAQ + 知识库）
│   ├── embedding.py        # 向量化
│   └── milvus_client.py    # Milvus 连接管理
├── voice/
│   ├── asr_service.py      # ASR（Whisper 本地）
│   ├── xunfei_asr.py       # 讯飞 ASR（云端）
│   └── tts_service.py      # 讯飞 TTS
├── avatar/driver.py        # 数字人动作驱动（配置化）
├── skill/
│   ├── loader.py           # Skill 加载器（代码 + Redis）
│   └── skills/             # 内置 Skill 定义
├── mcp/
│   ├── registry.py         # 工具注册表
│   └── sandbox.py          # 工具沙箱执行
└── infrastructure/         # Redis/PG 连接池
```

## 测试

```bash
pytest tests/ -q
```

P0 部署验证（需服务运行中）：

```bash
python scripts/verify_p0.py
```

## 配合前端使用

本服务不包含前端界面。你需要自行开发或使用配套的前端客户端：

- **大屏 Web 端**：React 应用，全屏数字人视频 + 语音交互 + 管理后台
- **Android 客户端**：一体机全屏模式 + 手机对话模式，通过同一 WS 协议对接
- **其他客户端**：任何能建立 WebSocket 连接的客户端均可对接，按上述协议收发消息即可

前端需要实现的核心能力：麦克风采集（PCM 16kHz）、WS 通信、TTS 音频播放、数字人视频/动画切换。

## License

Apache 2.0
