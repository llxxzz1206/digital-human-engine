from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

from app.config.settings import settings

logger = logging.getLogger(__name__)

# 讯飞建议：每 40ms 发送 1280 字节（16k/16bit/单声道）。
# 前端每 256ms 推一帧（8192 字节），这里在会话内再切成 1280B 小包喂给讯飞，
# 既符合官方分包规范，又无需改动前端。
FRAME_SIZE = 1280


class XunfeiAsrSession:
    """单条语音的讯飞流式识别会话。

    生命周期：start() 建连 → feed() 持续灌 PCM → finish() 收尾并等待最终结果 → close()。
    内部维护一个接收协程实时合并 wpgs 动态修正结果，finish() 只需等待结束帧。
    """

    def __init__(
        self,
        on_partial=None,
        eos: int | None = None,
        hotwords: str | None = None,
    ) -> None:
        self._app_id = settings.asr.app_id
        self._api_key = settings.asr.api_key
        self._api_secret = settings.asr.api_secret
        self._eos = eos if eos is not None else settings.asr.eos
        self._hotwords = hotwords if hotwords is not None else settings.asr.hotwords
        self._on_partial = on_partial  # async func(text: str) -> None，中间结果回调

        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._seq = 0                 # 已发送帧数（payload.audio.seq 从 1 开始）
        self._first_sent = False      # 是否已发首帧（首帧带 parameter）
        self._send_buf = bytearray()  # 待发送 PCM 缓冲（不足 1280B 暂存）
        self._result_map: dict[int, str] = {}  # sn -> 该片文本（wpgs 合并用）
        self._final_text = ""
        self._done = asyncio.Event()
        self._error: str | None = None

    # ── 鉴权 ──
    def _build_auth_url(self) -> str:
        url = "wss://iat.xf-yun.com/v1"
        date = format_date_time(mktime(datetime.now().timetuple()))
        signature_origin = (
            "host: iat.xf-yun.com\n"
            "date: " + date + "\n"
            "GET /v1 HTTP/1.1"
        )
        signature_sha = hmac.new(
            self._api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(signature_sha).decode("utf-8")
        authorization_origin = (
            'api_key="%s", algorithm="%s", headers="%s", signature="%s"'
            % (self._api_key, "hmac-sha256", "host date request-line", signature)
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
        params = {"authorization": authorization, "date": date, "host": "iat.xf-yun.com"}
        return url + "?" + urlencode(params)

    # ── 生命周期 ──
    async def start(self) -> None:
        import websockets

        if not self._app_id or not self._api_key or not self._api_secret:
            raise RuntimeError("讯飞 ASR 配置缺失: APP_ID/API_KEY/API_SECRET")

        auth_url = self._build_auth_url()
        self._ws = await websockets.connect(auth_url)
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("讯飞 ASR 会话已建立")

    async def feed(self, pcm: bytes) -> None:
        """灌入一段 PCM（16k/16bit/单声道）。内部按 1280B 分包发送。"""
        if not pcm:
            return
        self._send_buf.extend(pcm)
        while len(self._send_buf) >= FRAME_SIZE:
            chunk = bytes(self._send_buf[:FRAME_SIZE])
            del self._send_buf[:FRAME_SIZE]
            await self._send_frame(chunk, status=0 if not self._first_sent else 1)

    async def finish(self, timeout: float = 6.0) -> str:
        """收尾：flush 残余缓冲 + 发送结束帧，等待最终识别结果。

        超时则返回当前已合并的部分结果（不抛异常，保证链路不卡死）。
        """
        # flush 不足 1280B 的尾包
        if self._send_buf:
            chunk = bytes(self._send_buf)
            self._send_buf = bytearray()
            await self._send_frame(chunk, status=0 if not self._first_sent else 1)
        elif not self._first_sent:
            # 极端情况：一帧音频都没发过，仍需发首帧建会话
            await self._send_frame(b"", status=0)

        # 结束帧
        await self._send_frame(b"", status=2)

        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("讯飞 ASR 等待最终结果超时，使用部分结果: '%s'", self._final_text[:30])

        return self._final_text.strip()

    def partial_text(self) -> str:
        return self._final_text

    async def close(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._recv_task = None

    # ── 内部：发送 ──
    async def _send_frame(self, audio: bytes, status: int) -> None:
        self._seq += 1
        header: dict = {"app_id": self._app_id, "status": status}

        frame: dict = {"header": header}

        # 首帧携带服务参数
        if status == 0:
            iat: dict = {
                "domain": "slm",
                "language": "zh_cn",
                "accent": "mandarin",
                "eos": self._eos,
                "dwa": "wpgs",
                "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
            }
            # 会话级热词（dhw），格式 "utf-8;词1|词2"
            if self._hotwords:
                iat["dhw"] = "utf-8;" + self._hotwords
            frame["parameter"] = {"iat": iat}
            self._first_sent = True

        frame["payload"] = {
            "audio": {
                "encoding": "raw",
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "seq": self._seq,
                "status": status,
                "audio": base64.b64encode(audio).decode("utf-8") if audio else "",
            }
        }

        await self._ws.send(json.dumps(frame))

    # ── 内部：接收 ──
    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                resp = json.loads(message)
                header = resp.get("header", {})
                code = header.get("code", -1)
                if code != 0:
                    self._error = header.get("message", "")
                    logger.error("讯飞 ASR 错误: sid=%s, code=%s, message=%s",
                                 header.get("sid", ""), code, self._error)
                    self._done.set()
                    return

                result = resp.get("payload", {}).get("result")
                if result and result.get("text"):
                    self._merge_result(result["text"])
                    if self._on_partial:
                        try:
                            await self._on_partial(self._final_text)
                        except Exception:
                            pass

                if header.get("status") == 2:
                    self._done.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("讯飞 ASR 接收循环异常: %s", e)
            self._error = str(e)
            self._done.set()

    def _merge_result(self, text_b64: str) -> None:
        """合并 wpgs 动态修正结果。

        pgs=="apd"：本片追加 → map[sn]=本片文本；
        pgs=="rpl"：替换 rg[0..1] 范围 → 删除这些 sn 后写入 map[sn]；
        最终文本 = 按 sn 升序拼接。
        """
        try:
            data = json.loads(base64.b64decode(text_b64).decode("utf-8"))
        except Exception as e:
            logger.error("讯飞 ASR 结果解码失败: %s", e)
            return

        sn = data.get("sn", 0)
        pgs = data.get("pgs", "apd")

        # 本片文本：ws[].cw[].w 拼接
        piece = ""
        for ws_item in data.get("ws", []):
            for cw in ws_item.get("cw", []):
                piece += cw.get("w", "")

        if pgs == "rpl":
            rg = data.get("rg") or []
            if len(rg) == 2:
                for i in range(rg[0], rg[1] + 1):
                    self._result_map.pop(i, None)

        self._result_map[sn] = piece
        self._final_text = "".join(self._result_map[k] for k in sorted(self._result_map))


class XunfeiAsrManager:
    """管理多个并发语音会话的讯飞 ASR 连接（session_id -> XunfeiAsrSession）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, XunfeiAsrSession] = {}

    async def get_or_create(self, session_id: str, on_partial=None) -> XunfeiAsrSession:
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = XunfeiAsrSession(on_partial=on_partial)
            await sess.start()
            self._sessions[session_id] = sess
        return sess

    def get(self, session_id: str) -> XunfeiAsrSession | None:
        return self._sessions.get(session_id)

    async def remove(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess:
            await sess.close()


xunfei_asr_manager = XunfeiAsrManager()
