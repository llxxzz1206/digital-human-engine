"""全链路延迟标定探针。

用讯飞 TTS 合成若干导诊问句当作"用户语音"（PCM），通过 WebSocket 走完整条链路
（流式ASR → RAG → LLM → TTS），测量：
  - 客户端"说完(audio.end)→收到首个回复音频(tts.audio)"延迟（用户真实感知）
  - 服务端 /admin/timings 分阶段拆解（ASR/RAG/LLM/TTS）

用法：.venv/Scripts/python.exe _calibrate.py
"""
import asyncio
import base64
import json
import time
import uuid

import httpx
import websockets

from app.voice.tts_service import tts_service

WS_URL = "ws://localhost:8000/ws"
ADMIN_TIMINGS = "http://localhost:8000/admin/timings"

# 导诊常见问句（覆盖科室/楼层/专有词），合成后当用户语音
QUESTIONS = [
    "请问门诊楼怎么走？",
    "我想去二楼挂号，应该怎么走？",
    "急诊科在哪个位置？",
    "抽血化验在哪里做？",
    "CT室怎么走？",
]

CHUNK = 8192  # 256ms @16k/16bit/mono，与前端 audioRecorder 一致


async def synth_pcm(text: str) -> bytes:
    pcm = bytearray()
    async for chunk in tts_service.synthesize_stream(text):
        if chunk:
            pcm.extend(chunk)
    return bytes(pcm)


async def drain(ws, quiet_s: float = 1.5) -> None:
    """排空当前回合的残留消息（多句 TTS/avatar 等），静默 quiet_s 秒视为回合结束。"""
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=quiet_s)
        except asyncio.TimeoutError:
            return


async def run_one(ws, session_id: str, channel_id: str, user_id: str,
                  text: str, pcm: bytes) -> dict:
    # 流式送音频（模拟前端 256ms/帧）
    for i in range(0, len(pcm), CHUNK):
        chunk = pcm[i:i + CHUNK]
        await ws.send(json.dumps({
            "type": "audio.stream",
            "payload": {
                "sessionId": session_id,
                "channelId": channel_id,
                "userId": user_id,
                "audio": base64.b64encode(chunk).decode("ascii"),
                "index": i // CHUNK,
                "format": "pcm",
            },
        }))
        await asyncio.sleep(0.256)

    # 说完
    t_end = time.monotonic()
    await ws.send(json.dumps({
        "type": "audio.end",
        "payload": {
            "sessionId": session_id,
            "channelId": channel_id,
            "userId": user_id,
            "format": "pcm",
        },
    }))

    # 等首个回复音频（用户感知"出声"）
    first_tts_at = None
    asr_text = None
    asr_ms_server = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
        except asyncio.TimeoutError:
            break
        msg = json.loads(raw)
        mtype = msg.get("type")
        payload = msg.get("payload", {})
        if mtype == "asr.result" and payload.get("isFinal"):
            asr_text = payload.get("text")
            asr_ms_server = payload.get("asrMs")
        elif mtype == "tts.audio" and payload.get("audio") and first_tts_at is None:
            first_tts_at = time.monotonic()
            break  # 拿到首个回复音频即可

    perceived_ms = int((first_tts_at - t_end) * 1000) if first_tts_at else None

    # 排空本轮剩余消息（后续句子 TTS / avatar 等），保证下一轮干净
    await drain(ws)

    return {
        "question": text,
        "asr_text": asr_text,
        "asr_ms_server": asr_ms_server,
        "perceived_ms": perceived_ms,  # audio.end → 首个回复音频
    }


async def main():
    run_start_ms = int(time.time() * 1000)

    # 预合成所有问句音频（不计入链路耗时）
    print("[准备] 合成问句音频 ...")
    pcms = []
    for q in QUESTIONS:
        pcm = await synth_pcm(q)
        pcms.append(pcm)
        print(f"  {q} → {len(pcm)} bytes ({len(pcm)/32000:.2f}s)")

    session_id = f"cal-{uuid.uuid4().hex[:8]}"
    channel_id = "cal-ch"
    user_id = "cal-user"

    results = []
    async with websockets.connect(WS_URL, max_size=None) as ws:
        # 建会话
        await ws.send(json.dumps({
            "type": "session.create",
            "payload": {"sessionId": session_id, "userId": user_id,
                        "channelId": channel_id, "avatarId": "9a46e56af580"},
        }))
        await ws.recv()  # session.created

        # 挂载 hospital 技能（让 RAG 有知识库）
        await ws.send(json.dumps({
            "type": "skill.mount",
            "payload": {"sessionId": session_id, "skillId": "hospital",
                        "channelId": channel_id},
        }))
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except asyncio.TimeoutError:
            pass

        for q, pcm in zip(QUESTIONS, pcms):
            print(f"\n[轮] {q}")
            r = await run_one(ws, session_id, channel_id, user_id, q, pcm)
            results.append(r)
            print(f"  识别: {r['asr_text']}")
            print(f"  服务端ASR尾延迟: {r['asr_ms_server']}ms | 客户端说完→出声: {r['perceived_ms']}ms")
            await asyncio.sleep(1.0)  # 轮间冷却

    # 读服务端分阶段明细，只取本次运行的新鲜记录（ts>=run_start_ms）
    print("\n[服务端 /admin/timings 本次运行明细]")
    async with httpx.AsyncClient() as client:
        resp = await client.get(ADMIN_TIMINGS, params={"limit": 200})
        body = resp.json()
    records = body.get("data", body).get("records", [])
    fresh = [r for r in records if isinstance(r.get("ts"), (int, float)) and r["ts"] >= run_start_ms]
    print(f"  本次运行记录数: {len(fresh)}")

    def p50(vals):
        vals = sorted(vals)
        return vals[len(vals)//2] if vals else 0

    for field in ("asr", "ragTotal", "ragSearch", "ragRerank",
                  "llmTotal", "llmTtft", "ttsTotal", "e2e"):
        vals = [r[field] for r in fresh if isinstance(r.get(field), (int, float))]
        if vals:
            print(f"  {field:12s} n={len(vals):2d} avg={int(sum(vals)/len(vals)):5d}ms p50={p50(vals):5d}ms")
        else:
            print(f"  {field:12s} n= 0 (无数据)")

    # 客户端汇总
    print("\n[客户端 说完→出声 汇总]")
    vals = sorted(r["perceived_ms"] for r in results if r["perceived_ms"] is not None)
    if vals:
        p50 = vals[len(vals)//2]
        print(f"  样本={len(vals)} min={vals[0]}ms p50={p50}ms max={vals[-1]}ms")


if __name__ == "__main__":
    asyncio.run(main())
