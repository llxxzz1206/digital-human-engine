# Python AI Engine Dockerfile
# 多阶段构建：uv 安装依赖 → 运行时镜像

# ── 阶段1: 构建 ──
FROM python:3.11-slim AS builder

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先复制依赖文件（利用 Docker 缓存）
COPY pyproject.toml uv.lock ./

# 安装依赖到虚拟环境
RUN uv sync --frozen --no-dev --no-install-project

# 复制应用代码
COPY . .

# 安装项目本身
RUN uv sync --frozen --no-dev

# ── 阶段2: 运行时 ──
FROM python:3.11-slim AS runtime

# 安装 ffmpeg（Whisper ASR 依赖）和 curl（健康检查）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制应用代码
COPY --from=builder /app /app

# Whisper 模型缓存目录（避免每次启动重新下载）
VOLUME /app/.cache

# 环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
