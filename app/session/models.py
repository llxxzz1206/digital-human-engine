from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


# 平台类型
Platform = Literal["fixed_terminal", "mobile_app", "mini_app", "web_admin"]

# 用户类型（影响讲解风格）
UserType = Literal["general", "family", "student", "scholar"]

# 位置来源
LocationSource = Literal["", "gps", "beacon", "manual", "user_input"]


class Session(BaseModel):
    """会话模型，支持固定终端和移动端"""

    sessionId: str
    # 用户身份（移动端必填，固定终端可选）
    userId: str = Field(default="", description="用户ID，移动端必填")
    avatarId: str = ""
    mountedSkills: list[str] = []

    # 平台类型（区分固定终端和移动端）
    platform: Platform = Field(
        default="fixed_terminal",
        description="平台类型: fixed_terminal/mobile_app/mini_app"
    )

    # 场景与设备（固定终端）
    sceneId: str = ""
    deviceId: str = ""
    deviceLocation: str = ""  # 固定位置描述（如"2楼大厅"）

    # 移动端实时位置
    currentLocation: str = Field(
        default="",
        description="移动端实时位置（GPS推断，如'消化内科候诊区')"
    )
    locationSource: LocationSource = Field(
        default="",
        description="位置来源: gps/beacon/manual/user_input"
    )
    locationUpdatedAt: int = Field(
        default=0,
        description="位置更新时间戳（毫秒）"
    )

    # 用户类型（影响讲解风格）
    userType: UserType = Field(default="general", description="用户类型: general/family/student/scholar")

    # 客户端能力声明（决定后端是否下发音频/驱动指令）
    capabilities: list[str] = Field(
        default_factory=lambda: ["text", "audio", "avatar"],
        description="客户端能力: text/audio/avatar/push"
    )

    createdAt: int = 0
    updatedAt: int = 0
