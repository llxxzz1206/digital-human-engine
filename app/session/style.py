"""个性化讲解风格配置

根据用户类型调整讲解风格：
  - general（普通游客）：轻松、简洁、重点突出
  - family（亲子家庭）：故事化、生动、互动式
  - student（学生团）：知识性、启发式、可提问
  - scholar（专业学者）：学术化、深度、严谨
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StyleConfig:
    """讲解风格配置"""
    system_prompt_suffix: str  # 系统提示后缀
    tone_description: str      # 语调描述（用于前端展示）


# 风格配置表
STYLE_CONFIGS: dict[str, StyleConfig] = {
    "general": StyleConfig(
        system_prompt_suffix="\n\n讲解风格：轻松自然，重点突出，一般2-3句话概括，不超过50字。避免过多专业术语。",
        tone_description="轻松简洁",
    ),
    "family": StyleConfig(
        system_prompt_suffix="\n\n讲解风格：用故事化语言，多打比方，用'小朋友们'称呼。可以适当提问互动，比如'你们猜猜这是做什么用的？'。避免枯燥的年代数据。",
        tone_description="生动有趣",
    ),
    "student": StyleConfig(
        system_prompt_suffix="\n\n讲解风格：强调事实和知识，鼓励追问，可以提及历史背景和制作工艺。用'同学们'称呼，可以适当设置悬念引发思考。",
        tone_description="知识启发",
    ),
    "scholar": StyleConfig(
        system_prompt_suffix="\n\n讲解风格：学术化表述，引用出处，使用专业术语。可以深入讨论工艺细节、历史演变、文化意义。不作过度简化。",
        tone_description="学术深度",
    ),
}


def get_style_prompt(user_type: str) -> str:
    """获取用户类型对应的风格提示"""
    config = STYLE_CONFIGS.get(user_type, STYLE_CONFIGS["general"])
    return config.system_prompt_suffix


def get_style_description(user_type: str) -> str:
    """获取用户类型对应的风格描述"""
    config = STYLE_CONFIGS.get(user_type, STYLE_CONFIGS["general"])
    return config.tone_description


# 用户类型中文名映射
USER_TYPE_NAMES: dict[str, str] = {
    "general": "普通游客",
    "family": "亲子家庭",
    "student": "学生团体",
    "scholar": "专业学者",
}