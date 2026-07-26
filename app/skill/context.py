from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillContext:
    """Skill 运行时上下文，注入到工具执行环境"""

    session_id: str = ""
    skill_id: str = ""
    knowledge_collection: str = ""
    platform: str = "fixed_terminal"
    user_id: str = ""
    scene_id: str = ""
    device_location: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
