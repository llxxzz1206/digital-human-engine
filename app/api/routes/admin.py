"""管理后台 CRUD 接口 — 场景/设备(PostgreSQL) + 形象/Skill(Redis)"""
from __future__ import annotations

import logging
import uuid
import time

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.infrastructure.redis import RedisPool
from app.infrastructure.database import DatabasePool
from app.api.deps import verify_admin_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["管理后台"], dependencies=[Depends(verify_admin_token)])

# ── Models ──

class SceneBody(BaseModel):
    id: str | None = None
    code: str = ""
    name: str = ""
    description: str = ""
    icon: str | None = None
    status: str = "ENABLE"

class DeviceBody(BaseModel):
    id: str | None = None
    sceneId: str = ""
    code: str = ""
    name: str = ""
    location: str = ""
    skillIds: str = ""
    status: str = "ENABLE"

class AvatarBody(BaseModel):
    id: str | None = None
    name: str = ""
    scene: str = ""
    idleVideo: str = ""
    talkingVideo: str = ""
    greetingVideo: str = ""
    pointLeftVideo: str = ""
    pointRightVideo: str = ""
    bowVideo: str = ""
    # 渲染形态：video（视频形象）/ image2d（2D 分层形象）。缺省 video，老形象不受影响
    renderType: str = "video"
    # 形象专属唤醒暗号（如 ["你好小医"]）。空 = 前端用兜底暗号。存 Redis 时序列化为 JSON 字符串
    wakePhrases: list[str] = []
    # ── VAD 参数（前端语音活动检测，按形象可调）──
    vadMaxSegmentMs: int = 30000      # 单段说话最大时长（ms），超时截断
    vadSilenceMs: int = 800           # 说话中静音多久截断（ms）
    vadRmsThreshold: float = 0.02     # RMS 能量阈值（灵敏度），越小越灵敏
    vadListeningTimeout: int = 20000  # 无语音多久回 idle（ms）
    # ── 动作配置（前端可用手势列表，逗号分隔字符串）──
    gestures: str = ""       # 如 "idle,talking,greeting,point_left,point_right,bow,wave"，空=默认六态
    loopStates: str = ""     # 如 "idle,talking"，空=默认 idle+talking 循环
    status: str = "ENABLE"

class SkillBody(BaseModel):
    id: str | None = None
    name: str = ""
    description: str = ""
    scene: str = ""  # 关联场景
    tools: list[str] = []  # 工具列表
    knowledgeCollection: str = ""  # Milvus collection 名称
    systemPrompt: str = ""  # LLM 系统提示词
    status: str = "ENABLE"

class SkillToggleBody(BaseModel):
    skillId: str
    status: str = "ENABLE"

class SkillBuildBody(BaseModel):
    skillId: str

# ══════════════════════════════════════
# 场景管理 (PostgreSQL dh_scene)
# ══════════════════════════════════════

@router.get("/scene/list")
async def scene_list():
    pool = await DatabasePool.get()
    rows = await pool.fetch(
        "SELECT id, code, name, description, icon, sort_code, status, create_time "
        "FROM dh_scene WHERE delete_flag = 'NOT_DELETE' ORDER BY sort_code"
    )
    return {"code": 200, "data": [dict(r) for r in rows]}


@router.post("/scene/add")
async def scene_add(body: SceneBody):
    pool = await DatabasePool.get()
    sid = f"scene_{body.code}" if body.code else f"scene_{uuid.uuid4().hex[:8]}"
    await pool.execute(
        "INSERT INTO dh_scene (id, code, name, description, icon, sort_code, status, delete_flag, create_time) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,'NOT_DELETE',NOW())",
        sid, body.code, body.name, body.description, body.icon, 99, body.status,
    )
    return {"code": 200, "data": {"id": sid}}


@router.post("/scene/edit")
async def scene_edit(body: SceneBody):
    pool = await DatabasePool.get()
    await pool.execute(
        "UPDATE dh_scene SET name=$2, description=$3, icon=$4, status=$5 WHERE id=$1",
        body.id, body.name, body.description, body.icon, body.status,
    )
    return {"code": 200, "data": True}


@router.post("/scene/delete")
async def scene_delete(body: dict):
    pool = await DatabasePool.get()
    await pool.execute(
        "UPDATE dh_scene SET delete_flag='DELETED' WHERE id=$1", body.get("id", "")
    )
    return {"code": 200, "data": True}

# ══════════════════════════════════════
# 设备管理 (PostgreSQL dh_device)
# ══════════════════════════════════════

@router.get("/device/list")
async def device_list(sceneId: str = Query(default="")):
    pool = await DatabasePool.get()
    if sceneId:
        rows = await pool.fetch(
            "SELECT id, scene_id, code, name, location, skill_ids, sort_code, status, create_time "
            "FROM dh_device WHERE delete_flag='NOT_DELETE' AND scene_id=$1 ORDER BY sort_code", sceneId
        )
    else:
        rows = await pool.fetch(
            "SELECT id, scene_id, code, name, location, skill_ids, sort_code, status, create_time "
            "FROM dh_device WHERE delete_flag='NOT_DELETE' ORDER BY sort_code"
        )
    # 字段名转驼峰
    data = []
    for r in rows:
        d = dict(r)
        d["sceneId"] = d.pop("scene_id", "")
        d["skillIds"] = d.pop("skill_ids", "")
        d["sortCode"] = d.pop("sort_code", 0)
        d["createTime"] = str(d.pop("create_time", ""))
        data.append(d)
    return {"code": 200, "data": data}


@router.post("/device/add")
async def device_add(body: DeviceBody):
    pool = await DatabasePool.get()
    did = f"device_{uuid.uuid4().hex[:8]}"
    await pool.execute(
        "INSERT INTO dh_device (id, scene_id, code, name, location, skill_ids, sort_code, status, delete_flag, create_time) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'NOT_DELETE',NOW())",
        did, body.sceneId, body.code, body.name, body.location, body.skillIds, 99, body.status,
    )
    return {"code": 200, "data": {"id": did}}


@router.post("/device/edit")
async def device_edit(body: DeviceBody):
    pool = await DatabasePool.get()
    await pool.execute(
        "UPDATE dh_device SET name=$2, location=$3, skill_ids=$4, status=$5 WHERE id=$1",
        body.id, body.name, body.location, body.skillIds, body.status,
    )
    return {"code": 200, "data": True}


@router.post("/device/delete")
async def device_delete(body: dict):
    pool = await DatabasePool.get()
    await pool.execute(
        "UPDATE dh_device SET delete_flag='DELETED' WHERE id=$1", body.get("id", "")
    )
    return {"code": 200, "data": True}

# ══════════════════════════════════════
# 形象管理 (Redis Hash + 本地视频文件)
# ══════════════════════════════════════

import json
import shutil
import zipfile
from pathlib import Path

from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse

AVATAR_PREFIX = "digitalhuman:avatar:"
AVATAR_DIR = Path(__file__).parent.parent.parent.parent / "avatar-packages"


@router.get("/avatar/list")
async def avatar_list():
    r = await RedisPool.get()
    keys = []
    async for key in r.scan_iter(AVATAR_PREFIX + "*"):
        keys.append(key)
    data = []
    for key in keys:
        entries = await r.hgetall(key)
        if entries:
            # wakePhrases 存的是 JSON 字符串，还原为数组（老形象无此字段 → 空数组，前端走兜底暗号）
            if "wakePhrases" in entries:
                try:
                    entries["wakePhrases"] = json.loads(entries["wakePhrases"])
                except (ValueError, TypeError):
                    entries["wakePhrases"] = []
            else:
                entries["wakePhrases"] = []
            # VAD 参数：Redis 存 str → 还原为数值（老形象无此字段 → 用默认值）
            for field, default in [("vadMaxSegmentMs", 30000), ("vadSilenceMs", 800),
                                   ("vadListeningTimeout", 20000)]:
                try:
                    entries[field] = int(entries.get(field, default))
                except (ValueError, TypeError):
                    entries[field] = default
            try:
                entries["vadRmsThreshold"] = float(entries.get("vadRmsThreshold", 0.02))
            except (ValueError, TypeError):
                entries["vadRmsThreshold"] = 0.02
            data.append(entries)
    return {"code": 200, "data": data}


@router.post("/avatar/add")
async def avatar_add(body: AvatarBody):
    r = await RedisPool.get()
    aid = uuid.uuid4().hex[:12]
    mapping = {
        "id": aid, "name": body.name, "scene": body.scene,
        "idleVideo": body.idleVideo, "talkingVideo": body.talkingVideo,
        "greetingVideo": body.greetingVideo,
        "pointLeftVideo": body.pointLeftVideo, "pointRightVideo": body.pointRightVideo,
        "bowVideo": body.bowVideo,
        "renderType": body.renderType,
        "wakePhrases": json.dumps(body.wakePhrases, ensure_ascii=False),
        "vadMaxSegmentMs": str(body.vadMaxSegmentMs),
        "vadSilenceMs": str(body.vadSilenceMs),
        "vadRmsThreshold": str(body.vadRmsThreshold),
        "vadListeningTimeout": str(body.vadListeningTimeout),
        "gestures": body.gestures,
        "loopStates": body.loopStates,
        "status": body.status, "createTime": str(int(time.time() * 1000)),
    }
    await r.hset(AVATAR_PREFIX + aid, mapping=mapping)
    # 创建形象目录
    (AVATAR_DIR / aid / "videos").mkdir(parents=True, exist_ok=True)
    return {"code": 200, "data": {"id": aid}}


@router.post("/avatar/edit")
async def avatar_edit(body: AvatarBody):
    r = await RedisPool.get()
    if not body.id:
        return {"code": 400, "data": None}
    # exclude_unset：只更新请求里明确携带的字段。
    # 否则 Pydantic 会把未传的视频字段补成默认空串，把已上传的视频 URL 冲掉
    mapping = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None and k != "id"}
    # wakePhrases 是数组，Redis Hash 值必须是 str → 序列化为 JSON 字符串
    if isinstance(mapping.get("wakePhrases"), list):
        mapping["wakePhrases"] = json.dumps(mapping["wakePhrases"], ensure_ascii=False)
    mapping["updateTime"] = str(int(time.time() * 1000))
    await r.hset(AVATAR_PREFIX + body.id, mapping=mapping)
    return {"code": 200, "data": True}


@router.post("/avatar/delete")
async def avatar_delete(body: dict):
    r = await RedisPool.get()
    aid = body.get("id", "")
    await r.delete(AVATAR_PREFIX + aid)
    # 删除视频文件目录
    avatar_path = AVATAR_DIR / aid
    if avatar_path.exists():
        shutil.rmtree(avatar_path, ignore_errors=True)
    return {"code": 200, "data": True}


@router.post("/avatar/upload")
async def avatar_upload(
    avatarId: str = Form(...),
    state: str = Form(...),
    file: UploadFile = File(...),
):
    """上传形象视频文件（state: idle/talking/greeting/point_left/point_right/bow）"""
    field_map = {
        "idle": "idleVideo", "talking": "talkingVideo", "greeting": "greetingVideo",
        "point_left": "pointLeftVideo", "point_right": "pointRightVideo", "bow": "bowVideo",
    }
    if state not in field_map:
        return {"code": 400, "data": None, "msg": "state 必须为 idle/talking/greeting/point_left/point_right/bow"}

    video_dir = AVATAR_DIR / avatarId / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件（保留原始扩展名）
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    filename = f"{state}{ext}"
    filepath = video_dir / filename

    content = await file.read()
    filepath.write_bytes(content)

    # 更新 Redis 中对应状态的视频路径
    # URL 带版本号（毫秒时间戳）：重新上传后 URL 变化，强制浏览器/<video> 重新加载，
    # 否则同 URL 命中缓存，看起来像"替换没生效"
    r = await RedisPool.get()
    version = int(time.time() * 1000)
    await r.hset(
        AVATAR_PREFIX + avatarId,
        field_map[state],
        f"/api/admin/avatar/video/{avatarId}/{state}?v={version}",
    )

    return {"code": 200, "data": {"filename": filename, "size": len(content)}}


@router.get("/avatar/video/{avatar_id}/{state}")
async def avatar_video(avatar_id: str, state: str):
    """获取/预览形象视频文件"""
    video_dir = AVATAR_DIR / avatar_id / "videos"
    # 查找匹配的文件（可能是 .mp4/.webm/.mov）
    for f in video_dir.iterdir() if video_dir.exists() else []:
        if f.stem == state:
            return FileResponse(f, media_type="video/mp4")
    return {"code": 404, "data": None, "msg": "视频文件不存在"}


# ── 2D 分层形象资产（阶段2）──
# 约定式目录：avatar-packages/{id}/image2d/
#   config.json            嘴型数量 + 嘴部区域（相对底图的比例坐标）
#   base_{state}.{ext}     各状态底图，缺省回退 base_idle
#   mouth_{i}.{ext}        嘴型序列（i = 0..N，0 为闭合）
IMAGE2D_EXTS = [".png", ".svg", ".webp", ".jpg", ".jpeg"]

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png", ".svg": "image/svg+xml", ".webp": "image/webp",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
}


def _find_image(directory: Path, stem: str) -> Path | None:
    """按文件名主干在目录里找图片（扩展名不限，取白名单内第一个匹配）"""
    if not directory.exists():
        return None
    for ext in IMAGE2D_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


@router.get("/avatar/image2d/{avatar_id}/config")
async def avatar_image2d_config(avatar_id: str):
    """2D 形象配置（嘴型数量 + 嘴部区域）。不存在 → 404，前端据此回退视频形象"""
    cfg = AVATAR_DIR / avatar_id / "image2d" / "config.json"
    if not cfg.exists():
        return {"code": 404, "data": None, "msg": "该形象未配置 2D 资产"}
    return {"code": 200, "data": json.loads(cfg.read_text(encoding="utf-8"))}


@router.get("/avatar/image2d/{avatar_id}/base/{state}")
async def avatar_image2d_base(avatar_id: str, state: str):
    """状态底图。本状态无专属底图时回退 base_idle（与视频 resolvedField 回退链同思路）"""
    img_dir = AVATAR_DIR / avatar_id / "image2d"
    f = _find_image(img_dir, f"base_{state}") or _find_image(img_dir, "base_idle")
    if not f:
        return {"code": 404, "data": None, "msg": "底图不存在"}
    # no-cache：管理台重传底图后浏览器走协商缓存，避免渲染端拿到旧图
    return FileResponse(f, media_type=_IMAGE_MEDIA_TYPES.get(f.suffix.lower(), "image/png"),
                        headers={"Cache-Control": "no-cache"})


@router.get("/avatar/image2d/{avatar_id}/mouth/{index}")
async def avatar_image2d_mouth(avatar_id: str, index: int):
    """嘴型序列图（index 0 = 闭合）"""
    f = _find_image(AVATAR_DIR / avatar_id / "image2d", f"mouth_{index}")
    if not f:
        return {"code": 404, "data": None, "msg": "嘴型图不存在"}
    return FileResponse(f, media_type=_IMAGE_MEDIA_TYPES.get(f.suffix.lower(), "image/png"),
                        headers={"Cache-Control": "no-cache"})


def _remove_stem_variants(directory: Path, stem: str) -> None:
    """删掉同主干的旧扩展名变体，避免上传 .png 后 _find_image 仍命中旧 .svg"""
    if not directory.exists():
        return
    for ext in IMAGE2D_EXTS:
        old = directory / f"{stem}{ext}"
        if old.exists():
            old.unlink()


@router.post("/avatar/image2d/upload")
async def avatar_image2d_upload(
    avatarId: str = Form(...),
    kind: str = Form(...),            # base / mouth
    state: str = Form(""),            # kind=base 用：idle/talking/...
    index: int = Form(-1),            # kind=mouth 用：0..N
    file: UploadFile = File(...),
):
    """上传 2D 形象资产：底图（base_{state}）或嘴型序列图（mouth_{index}）"""
    img_dir = AVATAR_DIR / avatarId / "image2d"
    img_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "img.png").suffix.lower()
    if ext not in IMAGE2D_EXTS:
        return {"code": 400, "data": None, "msg": f"不支持的图片格式 {ext}"}

    if kind == "base":
        if state not in ("idle", "talking", "greeting", "point_left", "point_right", "bow"):
            return {"code": 400, "data": None, "msg": "state 必须为 idle/talking/greeting/point_left/point_right/bow"}
        stem = f"base_{state}"
    elif kind == "mouth":
        if index < 0:
            return {"code": 400, "data": None, "msg": "kind=mouth 时 index 必须 >= 0"}
        stem = f"mouth_{index}"
    else:
        return {"code": 400, "data": None, "msg": "kind 必须为 base/mouth"}

    _remove_stem_variants(img_dir, stem)
    filepath = img_dir / f"{stem}{ext}"
    content = await file.read()
    filepath.write_bytes(content)
    return {"code": 200, "data": {"filename": filepath.name, "size": len(content)}}


@router.post("/avatar/image2d/saveConfig")
async def avatar_image2d_save_config(body: dict):
    """保存 2D 形象配置（嘴型数量 + 嘴部区域比例坐标）→ config.json"""
    avatar_id = body.get("avatarId", "")
    mouth_shape_count = int(body.get("mouthShapeCount", 0))
    region = body.get("mouthRegion", {})
    if not avatar_id or mouth_shape_count <= 0:
        return {"code": 400, "data": None, "msg": "avatarId 与 mouthShapeCount 必填"}
    img_dir = AVATAR_DIR / avatar_id / "image2d"
    img_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mouthShapeCount": mouth_shape_count,
        "mouthRegion": {
            "x": float(region.get("x", 0)), "y": float(region.get("y", 0)),
            "w": float(region.get("w", 0)), "h": float(region.get("h", 0)),
        },
    }
    (img_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"code": 200, "data": config}


@router.get("/avatar/export/{avatar_id}")
async def avatar_export(avatar_id: str):
    """导出形象包为 zip（config.json + videos/）"""
    r = await RedisPool.get()
    entries = await r.hgetall(AVATAR_PREFIX + avatar_id)
    if not entries:
        return {"code": 404, "data": None, "msg": "形象不存在"}

    # 生成 config.json（基于实际存在的视频文件）
    video_dir = AVATAR_DIR / avatar_id / "videos"
    loop_states = {"idle", "talking"}
    states_cfg: dict = {}
    if video_dir.exists():
        for f in sorted(video_dir.iterdir()):
            state_name = f.stem
            entry: dict = {"file": f"videos/{f.name}"}
            if state_name in loop_states:
                entry["loop"] = True
            else:
                entry["loop"] = False
                entry["fallback"] = "idle"
            states_cfg[state_name] = entry

    config = {
        "name": entries.get("name", ""),
        "scene": entries.get("scene", ""),
        "states": states_cfg,
    }

    # 打包 zip
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(config, ensure_ascii=False, indent=2))
        if video_dir.exists():
            for f in video_dir.iterdir():
                zf.write(f, f"videos/{f.name}")
    buf.seek(0)

    name = entries.get("name", avatar_id)
    from urllib.parse import quote
    encoded_name = quote(f"{name}-avatar-package.zip")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )

# ══════════════════════════════════════
# Skill 管理 (Redis Hash)
# ══════════════════════════════════════

SKILL_PREFIX = "digitalhuman:skill:"

@router.get("/skill/list")
async def skill_list():
    r = await RedisPool.get()
    keys = []
    async for key in r.scan_iter(SKILL_PREFIX + "*"):
        keys.append(key)
    data = []
    for key in keys:
        entries = await r.hgetall(key)
        if entries:
            data.append(entries)
    return {"code": 200, "data": data}


@router.post("/skill/add")
async def skill_add(body: SkillBody):
    """添加 Skill（手动配置，与代码加载的 Skill 并存）"""
    r = await RedisPool.get()
    sid = body.id or uuid.uuid4().hex[:8]
    mapping = {
        "id": sid,
        "name": body.name,
        "description": body.description,
        "scene": body.scene,
        "tools": json.dumps(body.tools, ensure_ascii=False),
        "knowledgeCollection": body.knowledgeCollection,
        "systemPrompt": body.systemPrompt,
        "status": body.status,
        "source": "manual",  # 标记为手动创建（区别于代码加载）
        "createTime": str(int(time.time() * 1000)),
    }
    await r.hset(SKILL_PREFIX + sid, mapping=mapping)
    return {"code": 200, "data": {"id": sid}}


@router.post("/skill/edit")
async def skill_edit(body: SkillBody):
    """编辑 Skill"""
    if not body.id:
        return {"code": 400, "data": None, "msg": "缺少 id"}

    r = await RedisPool.get()
    mapping = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None and k != "id"}
    # tools 是数组，需要序列化
    if "tools" in mapping and isinstance(mapping["tools"], list):
        mapping["tools"] = json.dumps(mapping["tools"], ensure_ascii=False)
    mapping["updateTime"] = str(int(time.time() * 1000))
    await r.hset(SKILL_PREFIX + body.id, mapping=mapping)
    return {"code": 200, "data": True}


@router.post("/skill/delete")
async def skill_delete(body: dict):
    """删除 Skill（Redis + 内存联动卸载）"""
    r = await RedisPool.get()
    sid = body.get("id", "")
    if not sid:
        return {"code": 400, "data": None, "msg": "缺少 id"}
    await r.delete(SKILL_PREFIX + sid)
    # 联动卸载内存中的 Skill（代码定义或 Redis 加载的）
    from app.skill.loader import skill_loader
    skill_loader.unload_skill(sid)
    return {"code": 200, "data": True}


@router.post("/skill/toggle")
async def skill_toggle(body: SkillToggleBody):
    r = await RedisPool.get()
    await r.hset(SKILL_PREFIX + body.skillId, "status", body.status)
    return {"code": 200, "data": True}


@router.post("/skill/buildKnowledge")
async def skill_build(body: SkillBuildBody):
    r = await RedisPool.get()
    await r.hset(SKILL_PREFIX + body.skillId, "knowledgeStatus", "BUILDING")
    return {"code": 200, "data": True}


# ══════════════════════════════════════
# 性能监控 (Redis digitalhuman:timings)
# ══════════════════════════════════════

# 参与聚合统计的数值字段（与 interaction_graph._record_timing 的落库结构对应）
_TIMING_FIELDS = (
    "asr", "ragTotal", "ragNav", "ragSearch", "ragRerank",
    "llmTotal", "llmTtft", "ttsTotal", "workflow", "e2e",
)


def _pctl(sorted_vals: list[int], p: float) -> int:
    """已排序列表的百分位数（线性插值），空列表返回 0"""
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return int(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


@router.get("/timings")
async def timings_list(limit: int = Query(50, ge=1, le=200)):
    """分阶段耗时明细 + 聚合统计。

    records: 最近 limit 条（新→旧），每条含 ASR/RAG/LLM/TTS 各阶段毫秒数；
    stats:   全量（最近 200 条）每阶段的 count/avg/p95，用于判断瓶颈。
    """
    r = await RedisPool.get()
    raw = await r.lrange("digitalhuman:timings", 0, -1)

    records: list[dict] = []
    for item in raw:
        try:
            records.append(json.loads(item))
        except (ValueError, TypeError):
            continue  # 跳过脏数据

    # 聚合：对每个数值字段收集非空值 → count/avg/p95
    stats: dict[str, dict] = {}
    for field in _TIMING_FIELDS:
        vals = sorted(rec[field] for rec in records
                      if isinstance(rec.get(field), (int, float)))
        stats[field] = {
            "count": len(vals),
            "avg": int(sum(vals) / len(vals)) if vals else 0,
            "p95": _pctl(vals, 0.95),
        }

    # 当前管线"出处"配置（不含任何密钥），供前端展示各阶段用的模型/服务
    from app.config.settings import settings
    asr_cfg = settings.asr
    asr_tag = (f"whisper·{asr_cfg.model}·{asr_cfg.device}"
               if asr_cfg.provider == "whisper" else asr_cfg.provider)
    config = {
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "apiBase": settings.llm.api_base,
        },
        "asr": {
            "provider": asr_cfg.provider,
            "model": asr_cfg.model,
            "device": asr_cfg.device,
            "tag": asr_tag,
        },
        "tts": {
            "voice": settings.tts.voice,
            "speed": settings.tts.speed,
        },
        "embedding": {"model": settings.milvus.embedding_model},
        "rerank": {"model": settings.rag.rerank_model},
    }

    return {"code": 200, "data": {
        "records": records[:limit], "stats": stats, "total": len(records), "config": config,
    }}


# ── 观展报告 ──

@router.get("/visit/report/{session_id}")
async def visit_report(session_id: str):
    """生成观展报告：对话摘要 + 关注点 + 个性化标签 + 推荐路线
    
    Args:
        session_id: 会话ID
    
    Returns:
        报告JSON，包含 durationMinutes/conversationCount/topTopics/summary/tags/recommendations
    """
    from app.services.visit_report import generate_visit_report, report_to_dict
    
    report = await generate_visit_report(session_id)
    if report is None:
        return {"code": 404, "msg": "无对话记录", "data": None}
    
    return {"code": 200, "data": report_to_dict(report)}


# ══════════════════════════════════════
# 系统配置管理
# ══════════════════════════════════════

class LLMConfigBody(BaseModel):
    provider: str = "zhipu"
    model: str = "glm-4-flash"
    fast_model: str = ""
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 150

class ASRConfigBody(BaseModel):
    provider: str = "whisper"
    model: str = "small"
    hotwords: str = ""

class TTSConfigBody(BaseModel):
    voice: str = "x4_lingxiaoxuan_oral"
    speed: int = 50
    volume: int = 50

class RAGConfigBody(BaseModel):
    rerank_enabled: bool = False
    top_k: int = 3
    threshold_a: float = 0.85
    threshold_b: float = 0.5

class ConfigBody(BaseModel):
    category: str  # llm/asr/tts/rag
    config: dict


@router.get("/config/list")
async def config_list():
    """获取所有系统配置"""
    from app.services.system_config import system_config
    config = await system_config.get_config()
    return {"code": 200, "data": config}


@router.get("/config/{category}")
async def config_get(category: str):
    """获取指定类型的配置"""
    from app.services.system_config import system_config
    config = await system_config.get_config(category)
    return {"code": 200, "data": config}


@router.post("/config/update")
async def config_update(body: ConfigBody):
    """更新配置（支持热更新）"""
    from app.services.system_config import system_config
    success, message = await system_config.update_config(body.category, body.config)
    if success:
        return {"code": 200, "data": {"message": message}}
    else:
        return {"code": 400, "msg": message}


@router.post("/config/reset/{category}")
async def config_reset(category: str):
    """重置配置为默认值"""
    from app.services.system_config import system_config
    success = await system_config.reset_config(category)
    if success:
        return {"code": 200, "data": {"message": "配置已重置"}}
    else:
        return {"code": 400, "msg": "重置失败"}


@router.get("/config/history/{category}")
async def config_history(category: str, limit: int = Query(default=10)):
    """获取配置变更历史"""
    from app.services.system_config import system_config
    history = await system_config.get_change_history(category, limit)
    return {"code": 200, "data": history}


@router.post("/config/test-llm")
async def config_test_llm(body: LLMConfigBody):
    """测试 LLM 连接"""
    from app.services.system_config import system_config
    success, message = await system_config.test_llm_connection(
        body.provider, body.model, body.api_key, body.api_base
    )
    if success:
        return {"code": 200, "data": {"message": message}}
    else:
        return {"code": 400, "msg": message}
