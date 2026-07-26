from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import AsyncGenerator
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from time import mktime

from app.config.settings import settings

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self) -> None:
        self._app_id = settings.tts.app_id
        self._api_key = settings.tts.api_key
        self._api_secret = settings.tts.api_secret
        self._voice = settings.tts.voice
        self._speed = settings.tts.speed

    def _build_auth_url(self) -> str:
        """生成讯飞 TTS WebSocket 鉴权 URL（遵循官方 demo 格式）"""
        url = "wss://tts-api.xfyun.cn/v2/tts"

        # RFC1123 格式时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接签名字符串（注意 GET 后面的空格和 /v2/tts 后面的空格）
        signature_origin = "host: ws-api.xfyun.cn\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET /v2/tts HTTP/1.1"

        # hmac-sha256 加密
        signature_sha = hmac.new(
            self._api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = (
            "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\""
            % (self._api_key, "hmac-sha256", "host date request-line", signature_sha)
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(encoding="utf-8")

        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn",
        }
        return url + "?" + urlencode(v)

    def _build_request_frame(self, text: str, voice: str | None = None, speed: int | None = None) -> str:
        """构建讯飞 TTS 请求帧（遵循官方 common/business/data 结构）"""
        frame = {
            "common": {"app_id": self._app_id},
            "business": {
                "aue": "raw",
                "auf": "audio/L16;rate=16000",
                "vcn": voice or self._voice,
                "speed": speed or self._speed,
                "volume": 50,
                "pitch": 50,
                "tte": "utf8",
            },
            "data": {
                "status": 2,
                "text": str(base64.b64encode(text.encode("utf-8")), "UTF8"),
            },
        }
        return json.dumps(frame)

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """流式 TTS 合成（讯飞 WebSocket API）"""
        import websockets

        # TTS 未配置时返回空音频，不抛异常，让上层继续走文本流
        if not self._app_id or not self._api_key or not self._api_secret:
            logger.error("讯飞 TTS 配置缺失: APP_ID/API_KEY/API_SECRET")
            yield b""
            return

        auth_url = self._build_auth_url()
        request_frame = self._build_request_frame(text, voice, speed)

        logger.info("TTS 合成: text_len=%d, voice=%s", len(text), voice or self._voice)

        try:
            async with websockets.connect(auth_url) as ws:
                await ws.send(request_frame)

                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.error("TTS 接收超时: text_len=%d", len(text))
                        break
                    resp_data = json.loads(response)

                    code = resp_data.get("code", -1)
                    if code != 0:
                        message = resp_data.get("message", "")
                        sid = resp_data.get("sid", "")
                        logger.error("TTS 合成错误: sid=%s, code=%s, message=%s", sid, code, message)
                        break

                    # 音频数据在 data.audio 字段
                    audio_data = resp_data.get("data", {}).get("audio", "")
                    if audio_data:
                        yield base64.b64decode(audio_data)

                    status = resp_data.get("data", {}).get("status", 0)
                    if status == 2:
                        break

        except Exception as e:
            logger.error("TTS 合成失败: %s", e)
            yield b""


tts_service = TTSService()
