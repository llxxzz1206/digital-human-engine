"""观展报告生成服务：自动生成个性化参观报告

功能：
  - 从对话日志提取用户关注点
  - 用 LLM 生成摘要和推荐
  - 生成个性化标签（如"深度探索者"、"打卡达人"）

调用方式：
  GET /api/admin/visit/report/{session_id}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.llm.service import llm_service
from app.services.chat_logger import chat_logger

logger = logging.getLogger(__name__)


@dataclass
class VisitReport:
    """观展报告"""
    session_id: str
    duration_minutes: int
    conversation_count: int
    top_topics: list[str]  # 用户关注的展品/话题 Top3
    summary: str           # LLM 生成的参观摘要
    tags: list[str]        # 个性化标签
    recommendations: list[str]  # 推荐后续参观路线


async def generate_visit_report(session_id: str) -> VisitReport | None:
    """生成观展报告
    
    Args:
        session_id: 会话ID
    
    Returns:
        观展报告，如果无对话则返回 None
    """
    # 1. 读取对话日志
    records = chat_logger.read_logs(session_id=session_id)
    if not records:
        logger.info("Session %s 无对话记录，跳过报告生成", session_id)
        return None
    
    # 2. 统计基本信息
    first_ts = records[0].get("timestamp", "")
    last_ts = records[-1].get("timestamp", "")
    
    # 计算时长
    from datetime import datetime
    try:
        start = datetime.fromisoformat(first_ts)
        end = datetime.fromisoformat(last_ts)
        duration_minutes = int((end - start).total_seconds() / 60)
    except Exception:
        duration_minutes = 0
    
    conversation_count = len(records)
    
    # 3. 提取用户关注点（基于 rerank_score 和 rag_hits）
    topic_scores: dict[str, float] = {}
    for r in records:
        hits = r.get("rag_hits", [])
        score = r.get("rerank_score", 0)
        for hit in hits:
            if hit and score > 0.5:  # 只关注相关度高的命中
                topic = hit[:50]  # 截取前50字符作为主题
                topic_scores[topic] = max(topic_scores.get(topic, 0), score)
    
    # 按分数排序，取 Top3
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    top_topics = [t[0] for t in sorted_topics[:3]]
    
    # 4. 用 LLM 生成摘要和推荐
    user_inputs = [r.get("user_input", "") for r in records if r.get("user_input")]
    replies = [r.get("reply", "") for r in records if r.get("reply")]
    
    prompt = f"""你是一个博物馆导览助手。请根据用户的参观对话记录，生成一份参观报告。

用户问题列表：
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(user_inputs[:10]))}

助手回复列表：
{chr(10).join(f'{i+1}. {a}' for i, a in enumerate(replies[:10]))}

请生成：
1. 参观摘要（2-3句话，概括用户关注的内容）
2. 个性化标签（如"深度探索者"、"打卡达人"、"知识爱好者"，选1-2个）
3. 推荐后续参观路线（如有）

请用JSON格式返回：
{{"summary": "...", "tags": ["..."], "recommendations": ["..."]}}
"""
    
    try:
        response = await llm_service.chat([{"role": "user", "content": prompt}])
        import json
        report_data = json.loads(response)
        summary = report_data.get("summary", "")
        tags = report_data.get("tags", [])
        recommendations = report_data.get("recommendations", [])
    except Exception as e:
        logger.warning("LLM 生成报告失败: %s，使用默认", e)
        # 降级方案：简单摘要
        summary = f"本次参观共进行了{conversation_count}次对话，关注了{len(top_topics)}个主题。"
        tags = ["参观者"]
        recommendations = []
    
    return VisitReport(
        session_id=session_id,
        duration_minutes=duration_minutes,
        conversation_count=conversation_count,
        top_topics=top_topics,
        summary=summary,
        tags=tags,
        recommendations=recommendations,
    )


def report_to_dict(report: VisitReport) -> dict[str, Any]:
    """转换报告为字典格式"""
    return {
        "sessionId": report.session_id,
        "durationMinutes": report.duration_minutes,
        "conversationCount": report.conversation_count,
        "topTopics": report.top_topics,
        "summary": report.summary,
        "tags": report.tags,
        "recommendations": report.recommendations,
    }