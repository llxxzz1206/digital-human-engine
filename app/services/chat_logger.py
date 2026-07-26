from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 日志根目录：项目根/logs/chat/
_LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "chat"


class ChatLogger:
    """对话日记服务 — 双格式写入

    - .jsonl: 机器可读，每条一行 JSON（供 API 查询）
    - .md: 人类可读，对话报告（供开发者直接查看）
    每天一对文件，如 2026-07-18.jsonl + 2026-07-18.md
    """

    def __init__(self) -> None:
        self._log_dir = _LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        session_id: str,
        user_input: str,
        reply: str,
        route: str,
        rerank_score: float = 0.0,
        rag_hits: list[str] | None = None,
        latency_ms: int = 0,
        skill_ids: list[str] | None = None,
        asr_text: str | None = None,
    ) -> None:
        """写入一条对话日记（JSONL + Markdown 双格式）"""
        now = datetime.now()
        record = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "user_input": user_input,
            "reply": reply,
            "route": route,
            "rerank_score": round(rerank_score, 4),
            "rag_hits": rag_hits or [],
            "latency_ms": latency_ms,
            "skill_ids": skill_ids or [],
            "asr_text": asr_text,
        }

        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # 1. 写入 JSONL（机器可读）
        jsonl_path = self._log_dir / f"{date_str}.jsonl"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("对话日记 JSONL 写入失败: %s", e)

        # 2. 追加写入 Markdown（人类可读）
        md_path = self._log_dir / f"{date_str}.md"
        try:
            self._append_markdown(md_path, time_str, record)
        except Exception as e:
            logger.error("对话日记 Markdown 写入失败: %s", e)

    def _append_markdown(self, md_path: Path, time_str: str, record: dict) -> None:
        """追加一条对话到 Markdown 报告"""
        # 如果文件不存在，先写表头
        if not md_path.exists():
            header = f"# 对话日记 {record['timestamp'][:10]}\n\n"
            header += "| 时间 | 路由 | 分数 | 延迟 | 用户输入 | 回复 | 语音原文 |\n"
            header += "|------|------|------|------|----------|------|----------|\n"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(header)

        route = record["route"]
        score = record["rerank_score"]
        latency = record["latency_ms"]
        user_input = record["user_input"].replace("|", "｜").replace("\n", " ")
        reply = record["reply"].replace("|", "｜").replace("\n", " ")
        asr_text = (record.get("asr_text") or "").replace("|", "｜").replace("\n", " ")

        # 截断过长的文本
        if len(user_input) > 40:
            user_input = user_input[:37] + "..."
        if len(reply) > 50:
            reply = reply[:47] + "..."

        asr_col = asr_text[:20] + "..." if len(asr_text) > 20 else asr_text

        row = f"| {time_str} | {route} | {score:.2f} | {latency}ms | {user_input} | {reply} | {asr_col} |\n"
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(row)

    def read_logs(
        self,
        date: str | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """读取对话日记（JSONL 格式）"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        filepath = self._log_dir / f"{date}.jsonl"
        if not filepath.exists():
            return []

        records = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if session_id and record.get("session_id") != session_id:
                        continue
                    records.append(record)
        except Exception as e:
            logger.error("对话日记读取失败: %s", e)

        # 最新在前（JSONL 按时间追加），后台分页第一页即最新对话
        records.reverse()
        return records

    def list_dates(self) -> list[str]:
        """列出有日志的日期"""
        dates = set()
        for f in self._log_dir.glob("*.jsonl"):
            dates.add(f.stem)
        return sorted(dates)


chat_logger = ChatLogger()
