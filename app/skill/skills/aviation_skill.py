"""航空科技展馆导览 Skill — 展区导航、展品讲解、互动体验引导"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.mcp.decorators import mcp_tool
from app.skill.base import SkillBase

logger = logging.getLogger(__name__)

# 展区数据（唯一权威数据源：data/aviation_zones.json）
_ZONES_FILE = Path(__file__).resolve().parents[3] / "data" / "aviation_zones.json"


def _load_zones() -> dict[str, dict]:
    """加载展区数据：{展区名: {floor, area, description, exhibits}}"""
    try:
        with open(_ZONES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {item["name"]: item for item in data["zones"]}
    except FileNotFoundError:
        logger.warning("展区数据文件不存在: %s，导航功能不可用", _ZONES_FILE)
        return {}


ZONES: dict[str, dict] = _load_zones()

# 展品→展区反向索引
_EXHIBIT_TO_ZONE: dict[str, str] = {}
for zone_name, zone_info in ZONES.items():
    for exhibit in zone_info.get("exhibits", []):
        _EXHIBIT_TO_ZONE[exhibit] = zone_name


# 展区→方向映射（按展馆实际布局）
_ZONE_DIRECTIONS: dict[str, str] = {
    "航空历史厅": "point_right",
    "飞行器模型区": "point_right",
    "航天探索厅": "point_left",
    "飞行模拟体验区": "point_left",
    "无人机科技区": "point_right",
    "儿童航空乐园": "point_left",
}


def _zone_to_gesture(zone_name: str) -> str:
    """根据展区判断指示方向"""
    return _ZONE_DIRECTIONS.get(zone_name, "point_right")


# 问句修饰词剥离
_QUESTION_WORDS = (
    "在哪里", "在哪儿", "在什么", "哪个区", "哪个厅", "怎么走", "怎么过去", "怎么去",
    "在哪", "哪儿", "位置", "请问", "哪里", "哪个展",
)


def _extract_keyword(query: str) -> str:
    """从问句中提取关键词"""
    core = query.strip()
    for word in _QUESTION_WORDS:
        core = core.replace(word, "")
    return core.strip()


class AviationSkill(SkillBase):
    name = "aviation"
    description = "航空科技展馆智能导览：展区导航、展品讲解、互动体验引导"
    tools = ["aviation_navigate", "aviation_exhibit_info"]
    knowledge_collection = "aviation_knowledge"
    system_prompt = (
        "你是一座航空科技展馆的AI数字人导览助手，部署在展馆大厅的智能终端上。\n\n"
        "你的身份：\n"
        "- 你是AI数字人导览员，固定在终端上，无法移动\n"
        "- 你通过语音和屏幕为游客提供服务\n\n"
        "你的职责：\n"
        "1. 为游客提供展区导航、展品位置查询\n"
        "2. 讲解航空科技知识（飞行原理、航天历史、无人机技术等）\n"
        "3. 推荐参观路线和互动体验项目\n"
        "4. 解答展品相关问题\n\n"
        "【重要】工具使用规则：\n"
        "- 只要用户询问任何展区、展品的位置（如'XX在哪''XX怎么走'），"
        "必须先调用 aviation_navigate 工具查询，禁止凭记忆直接回答位置。\n"
        "- 查询展品详细信息时调用 aviation_exhibit_info 工具。\n\n"
        "回答规范：\n"
        "- 讲解生动有趣，适当补充航空科技背后的故事\n"
        "- 回答简洁自然，通常2-3句话\n"
        "- 不使用emoji\n"
        "- 如果用户输入明显无意义或语音识别错误，回复'不好意思没听清，请再说一遍'\n"
    )

    async def on_mount(self, session_id: str) -> None:
        logger.info("AviationSkill 挂载到会话: %s", session_id)

    async def on_unmount(self, session_id: str) -> None:
        logger.info("AviationSkill 从会话卸载: %s", session_id)


# ── 工具实现 ──


@mcp_tool(
    name="aviation_navigate",
    description="查询航空展馆内展区/展品的位置，为游客提供导航指引",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要查询的展区或展品名称，如'航天探索厅'、'歼-20模型'、'飞行模拟器'",
            },
        },
        "required": ["query"],
    },
)
async def aviation_navigate_handler(arguments: dict, context: dict | None = None) -> dict:
    query = arguments.get("query", "").strip()
    if not query:
        return {"success": False, "message": "请提供要查询的展区或展品名称"}

    # 精确匹配展区名
    if query in ZONES:
        zone = ZONES[query]
        gesture = _zone_to_gesture(query)
        return {
            "success": True,
            "zone": query,
            "floor": zone.get("floor", ""),
            "area": zone.get("area", ""),
            "message": f"{query}位于{zone.get('floor', '')}{zone.get('area', '')}。",
            "gesture": gesture,
        }

    # 剥离问句修饰词
    core = _extract_keyword(query)

    # 精确匹配（提取后）
    if core and core != query and core in ZONES:
        zone = ZONES[core]
        gesture = _zone_to_gesture(core)
        return {
            "success": True,
            "zone": core,
            "floor": zone.get("floor", ""),
            "area": zone.get("area", ""),
            "message": f"{core}位于{zone.get('floor', '')}{zone.get('area', '')}。",
            "gesture": gesture,
        }

    # 展品→展区查找
    exhibit_zone = _EXHIBIT_TO_ZONE.get(query) or _EXHIBIT_TO_ZONE.get(core, "")
    if exhibit_zone:
        zone = ZONES[exhibit_zone]
        gesture = _zone_to_gesture(exhibit_zone)
        return {
            "success": True,
            "zone": exhibit_zone,
            "exhibit": query if query in _EXHIBIT_TO_ZONE else core,
            "floor": zone.get("floor", ""),
            "area": zone.get("area", ""),
            "message": f"'{query}'在{exhibit_zone}，位于{zone.get('floor', '')}{zone.get('area', '')}。",
            "gesture": gesture,
        }

    # 模糊匹配
    matches = []
    search_term = core or query
    for zone_name, zone_info in ZONES.items():
        if search_term in zone_name or zone_name in search_term:
            matches.append({"zone": zone_name, "floor": zone_info.get("floor", ""), "area": zone_info.get("area", "")})
            continue
        for exhibit in zone_info.get("exhibits", []):
            if search_term in exhibit or exhibit in search_term:
                matches.append({"zone": zone_name, "exhibit": exhibit, "floor": zone_info.get("floor", ""), "area": zone_info.get("area", "")})
                break

    if matches:
        gesture = _zone_to_gesture(matches[0]["zone"])
        zone_list = "、".join(f"{m['zone']}({m.get('floor', '')})" for m in matches[:5])
        return {
            "success": True,
            "matches": matches[:5],
            "message": f"找到相关展区：{zone_list}",
            "gesture": gesture,
        }

    # 未找到
    all_zones = {name: f"{info.get('floor', '')} {info.get('area', '')}" for name, info in ZONES.items()}
    return {
        "success": False,
        "message": f"未找到'{query}'的位置，建议到入口导览台咨询。以下是全部展区供参考。",
        "all_zones": all_zones,
    }


@mcp_tool(
    name="aviation_exhibit_info",
    description="查询航空展品的详细信息（历史背景、技术参数、趣闻等）",
    input_schema={
        "type": "object",
        "properties": {
            "exhibit_name": {
                "type": "string",
                "description": "展品名称，如'歼-20'、'长征五号'、'翼龙无人机'",
            },
        },
        "required": ["exhibit_name"],
    },
)
async def aviation_exhibit_info_handler(arguments: dict, context: dict | None = None) -> dict:
    exhibit_name = arguments.get("exhibit_name", "").strip()
    if not exhibit_name:
        return {"success": False, "message": "请提供展品名称"}

    # 查找展品所属展区
    zone_name = _EXHIBIT_TO_ZONE.get(exhibit_name, "")
    if not zone_name:
        # 模糊匹配
        for name in _EXHIBIT_TO_ZONE:
            if exhibit_name in name or name in exhibit_name:
                zone_name = _EXHIBIT_TO_ZONE[name]
                exhibit_name = name
                break

    if not zone_name:
        return {
            "success": False,
            "message": f"未找到'{exhibit_name}'的展品信息，建议咨询现场工作人员。",
        }

    zone = ZONES.get(zone_name, {})
    return {
        "success": True,
        "exhibit": exhibit_name,
        "zone": zone_name,
        "floor": zone.get("floor", ""),
        "message": f"'{exhibit_name}'位于{zone_name}（{zone.get('floor', '')}），欢迎前往参观。",
    }
