"""音频增强模块：噪音抑制 + VAD检测（展馆嘈杂环境优化）

层级处理：
  1. WebRTC VAD 检测语音段（过滤纯噪音帧）
  2. Spectral Gate 降噪（抑制背景噪音）
  3. 归一化输出（确保音量一致）

目标：在 70-85dB 噪音环境下，ASR 识别率从 ~60% 提升到 ~90%
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# 默认参数
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # WebRTC VAD 支持 10/20/30ms


@dataclass
class EnhancementConfig:
    """音频增强参数"""
    # VAD 灵敏度 (0-3, 0最激进过滤)
    vad_aggressiveness: int = 0  # 默认最不激进，避免过滤用户语音\n
    # 降噪强度 (0-1, 0不降噪)
    noise_reduce_strength: float = 0.7
    # 最小语音时长 (ms)，低于此视为噪音
    min_speech_duration_ms: int = 200
    # 是否启用降噪（可配置关闭以降低延迟）
    enable_denoise: bool = True


class AudioEnhancer:
    """音频增强器：VAD + 降噪 + 归一化
    
    使用方式：
        enhancer = AudioEnhancer()
        clean_audio = enhancer.process(raw_pcm_bytes)
    """
    
    def __init__(self, config: EnhancementConfig | None = None) -> None:
        self._config = config or EnhancementConfig()
        self._vad = None
        self._denoise_available = False
        
        # 延迟加载 VAD
        self._vad_loaded = False
        self._denoise_loaded = False
        
        # 统计
        self._total_frames = 0
        self._speech_frames = 0
    
    def _load_vad(self) -> None:
        """延迟加载 WebRTC VAD"""
        if self._vad_loaded:
            return
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._config.vad_aggressiveness)
            self._vad_loaded = True
            logger.info("WebRTC VAD 已加载: aggressiveness=%d", self._config.vad_aggressiveness)
        except ImportError:
            logger.warning("webrtcvad 未安装，VAD 检测禁用")
            self._vad_loaded = True
    
    def _load_denoise(self) -> None:
        """延迟加载降噪模块"""
        if self._denoise_loaded:
            return
        try:
            import noisereduce
            self._denoise_available = True
            logger.info("noisereduce 已加载")
        except ImportError:
            logger.warning("noisereduce 未安装，降噪禁用")
        self._denoise_loaded = True
    
    def process(self, pcm_bytes: bytes) -> bytes:
        """处理音频：VAD → 降噪 → 归一化
        
        Args:
            pcm_bytes: 原始 PCM 字节 (16kHz/16bit/单声道)
        
        Returns:
            处理后的 PCM 字节（可能为空，表示纯噪音）
        """
        if not pcm_bytes or len(pcm_bytes) < 2:
            return pcm_bytes
        
        # 1. VAD 检测（快速，延迟 <1ms）
        if self._config.vad_aggressiveness >= 0:
            self._load_vad()
            if self._vad and not self._is_speech(pcm_bytes):
                self._total_frames += 1
                logger.debug("VAD 判定非语音，跳过: len=%d", len(pcm_bytes))
                return b""  # 返回空表示噪音帧
            self._speech_frames += 1
        self._total_frames += 1
        
        # 2. 降噪（较慢，延迟 10-30ms）
        if self._config.enable_denoise and self._config.noise_reduce_strength > 0:
            pcm_bytes = self._denoise(pcm_bytes)
        
        # 3. 归一化
        pcm_bytes = self._normalize(pcm_bytes)
        
        return pcm_bytes
    
    def _is_speech(self, pcm_bytes: bytes) -> bool:
        """WebRTC VAD 检测是否为语音"""
        if not self._vad:
            return True
        
        # WebRTC VAD 要求帧长为 10/20/30ms
        frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # 16bit = 2 bytes
        
        # 如果音频短于一帧，直接返回 True（保守策略）
        if len(pcm_bytes) < frame_size:
            return True
        
        # 检查每帧，任一帧为语音即返回 True
        try:
            for i in range(0, len(pcm_bytes) - frame_size + 1, frame_size):
                frame = pcm_bytes[i:i + frame_size]
                if self._vad.is_speech(frame, SAMPLE_RATE):
                    return True
            return False
        except Exception as e:
            logger.warning("VAD 检测异常: %s", e)
            return True  # 异常时保守返回 True
    
    def _denoise(self, pcm_bytes: bytes) -> bytes:
        """频谱门限降噪"""
        if not self._denoise_available:
            self._load_denoise()
        
        if not self._denoise_available:
            return pcm_bytes
        
        try:
            import noisereduce as nr
            
            # PCM bytes → numpy array
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 降噪
            reduced = nr.reduce_noise(
                y=samples,
                sr=SAMPLE_RATE,
                prop_decrease=self._config.noise_reduce_strength,
                stationary=False,  # 非平稳噪音（展厅人声）
            )
            
            # numpy array → PCM bytes
            reduced_int = (reduced * 32768.0).clip(-32768, 32767).astype(np.int16)
            return reduced_int.tobytes()
        
        except Exception as e:
            logger.warning("降噪异常: %s", e)
            return pcm_bytes
    
    def _normalize(self, pcm_bytes: bytes) -> bytes:
        """音量归一化（防止过大或过小）"""
        if len(pcm_bytes) < 2:
            return pcm_bytes
        
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # 计算 RMS
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        if rms < 100:  # 静音
            return pcm_bytes
        
        # 归一化到 -12dB (0.25)
        target_rms = 0.25 * 32768.0
        if rms > 0:
            gain = target_rms / rms
            gain = min(gain, 4.0)  # 限制增益
            samples = (samples.astype(np.float32) * gain).clip(-32768, 32767).astype(np.int16)
        
        return samples.tobytes()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_frames": self._total_frames,
            "speech_frames": self._speech_frames,
            "noise_frames": self._total_frames - self._speech_frames,
            "speech_ratio": self._speech_frames / self._total_frames if self._total_frames > 0 else 0,
        }


# 全局实例（延迟初始化）
_audio_enhancer: AudioEnhancer | None = None


def get_audio_enhancer(config: EnhancementConfig | None = None) -> AudioEnhancer:
    """获取全局音频增强器实例"""
    global _audio_enhancer
    if _audio_enhancer is None:
        _audio_enhancer = AudioEnhancer(config)
    return _audio_enhancer
