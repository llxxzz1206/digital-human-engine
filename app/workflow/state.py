from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    """LangGraph 工作流状态定义"""

    # 基础信息
    session_id: str
    user_input: str
    skill_ids: list[str]

    # 平台类型（区分固定终端和移动端）
    platform: str  # fixed_terminal/mobile_app/mini_app/web_admin

    # 客户端能力声明（决定是否下发音频/驱动指令）
    capabilities: list[str]  # ["text", "audio", "avatar", "push"]

    # 场景与设备（多场景多设备架构）
    scene_id: str
    device_id: str
    device_location: str

    # 移动端实时位置
    currentLocation: str  # 实时位置描述
    locationSource: str   # 位置来源

    # 用户类型（影响讲解风格）
    user_type: str

    # RAG 检索结果 + 重排
    context: list[dict]
    max_rerank_score: float  # Reranker 最高分，用于双阈值路由
    route: str  # 路由结果: faq_direct/rag_chat/chat
    source_type: str  # 检索来源: faq/knowledge

    # 工具调用
    available_tools: list[dict]
    tool_calls: list[dict]
    tool_results: list[dict]

    # 对话
    messages: list[dict]
    reply: str
    asr_text: str  # ASR 识别原文（语音输入时）

    # Avatar 驱动
    drive_data: dict
    gesture: str  # 手势提示 (point_left/point_right/bow)，由 Skill 工具设置

    # 控制标志
    interrupted: bool

    # FAQ 候选记录
    faq_candidate_recorded: bool

    # 跨设备上下文
    cross_device_context: str  # 注入 LLM prompt 的跨设备上下文文本

    # 分阶段耗时（ms）：各节点写入自己那段，结束时汇总输出 [TIMING] 日志
    # 形如 {"asr": 120, "rag": {"total":..,"search":..,"rerank":..}, "llm": {"ttft":..,"total":..}, "tts": {"total":..,"sentences":..,"cache_hits":..}}
    timings: dict
