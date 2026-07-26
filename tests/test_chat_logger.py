"""对话日记读取顺序测试

后台日志表分页 20 条/页，必须最新在前，否则新对话藏在最后一页
（用户实测踩坑：刚对话完刷新页面看不到，因为在第 3 页）。
"""

import json

from app.services.chat_logger import chat_logger


def test_read_logs_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_logger, "_log_dir", tmp_path)
    records = [
        {"timestamp": "2026-07-21T10:00:00", "session_id": "s1", "user_input": "第一条"},
        {"timestamp": "2026-07-21T10:30:00", "session_id": "s1", "user_input": "第二条"},
        {"timestamp": "2026-07-21T10:54:00", "session_id": "s1", "user_input": "最新一条"},
    ]
    with open(tmp_path / "2026-07-21.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    result = chat_logger.read_logs(date="2026-07-21")
    assert [r["user_input"] for r in result] == ["最新一条", "第二条", "第一条"]
