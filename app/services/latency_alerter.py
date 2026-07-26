"""延迟告警服务 - 检测性能瓶颈并发送告警"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LatencyAlerter:
    """延迟告警器
    
    功能：
    1. 检测端到端延迟超过阈值（默认 3000ms）
    2. 记录告警日志
    3. 可扩展：支持 Webhook/邮件通知
    
    使用方法：
    ```python
    alerter = LatencyAlerter()
    await alerter.check(session_id, timings)
    ```
    """
    
    # 延迟阈值配置（毫秒）
    E2E_WARNING_THRESHOLD = 3000  # 端到端警告阈值
    E2E_CRITICAL_THRESHOLD = 5000  # 端到端严重阈值
    
    # 分阶段阈值
    STAGE_THRESHOLDS = {
        "asr": {"warning": 500, "critical": 1000},
        "rag": {"warning": 300, "critical": 800},
        "llm": {"warning": 1500, "critical": 3000},
        "tts": {"warning": 800, "critical": 2000},
    }
    
    def __init__(self) -> None:
        self._alert_counts: dict[str, int] = {}
        self._last_alert_time: dict[str, float] = {}
    
    async def check(
        self,
        session_id: str,
        timings: dict[str, Any],
        route: str = "chat",
    ) -> dict[str, Any]:
        """检查延迟并生成告警
        
        Args:
            session_id: 会话 ID
            timings: 分阶段耗时（ms）
            route: 路由类型
            
        Returns:
            告警信息（无告警时返回空字典）
        """
        alerts = []
        
        # 1. 检查端到端延迟
        e2e = timings.get("e2e", 0)
        if e2e >= self.E2E_CRITICAL_THRESHOLD:
            alerts.append({
                "level": "CRITICAL",
                "stage": "e2e",
                "value": e2e,
                "threshold": self.E2E_CRITICAL_THRESHOLD,
                "message": f"端到端延迟严重超标: {e2e}ms (阈值: {self.E2E_CRITICAL_THRESHOLD}ms)",
            })
        elif e2e >= self.E2E_WARNING_THRESHOLD:
            alerts.append({
                "level": "WARNING",
                "stage": "e2e",
                "value": e2e,
                "threshold": self.E2E_WARNING_THRESHOLD,
                "message": f"端到端延迟超标: {e2e}ms (阈值: {self.E2E_WARNING_THRESHOLD}ms)",
            })
        
        # 2. 检查各阶段延迟
        for stage, thresholds in self.STAGE_THRESHOLDS.items():
            stage_time = timings.get(stage, {})
            if isinstance(stage_time, dict):
                value = stage_time.get("total", 0)
            else:
                value = stage_time
            
            if value >= thresholds["critical"]:
                alerts.append({
                    "level": "CRITICAL",
                    "stage": stage,
                    "value": value,
                    "threshold": thresholds["critical"],
                    "message": f"{stage.upper()} 延迟严重超标: {value}ms (阈值: {thresholds['critical']}ms)",
                })
            elif value >= thresholds["warning"]:
                alerts.append({
                    "level": "WARNING",
                    "stage": stage,
                    "value": value,
                    "threshold": thresholds["warning"],
                    "message": f"{stage.upper()} 延迟超标: {value}ms (阈值: {thresholds['warning']}ms)",
                })
        
        # 3. 记录告警日志
        if alerts:
            for alert in alerts:
                logger.warning(
                    "[LATENCY_ALERT] session=%s, route=%s, level=%s, stage=%s, value=%dms, threshold=%dms",
                    session_id, route, alert["level"], alert["stage"], alert["value"], alert["threshold"]
                )
            
            # 更新告警计数
            key = f"{session_id}:{route}"
            self._alert_counts[key] = self._alert_counts.get(key, 0) + 1
            self._last_alert_time[key] = time.time()
            
            # 可选：发送外部告警（Webhook/邮件）
            # await self._send_external_alert(session_id, alerts)
        
        return {
            "session_id": session_id,
            "route": route,
            "e2e": e2e,
            "alerts": alerts,
            "alert_count": len(alerts),
        }
    
    def get_stats(self) -> dict[str, Any]:
        """获取告警统计"""
        return {
            "total_alerts": sum(self._alert_counts.values()),
            "alert_counts": self._alert_counts.copy(),
            "last_alert_time": self._last_alert_time.copy(),
        }
    
    async def _send_external_alert(self, session_id: str, alerts: list[dict]) -> None:
        """发送外部告警（可扩展）
        
        示例：Webhook 通知
        ```python
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://your-webhook-url",
                json={
                    "session_id": session_id,
                    "alerts": alerts,
                    "timestamp": time.time(),
                }
            )
        ```
        """
        pass


# 全局实例
latency_alerter = LatencyAlerter()