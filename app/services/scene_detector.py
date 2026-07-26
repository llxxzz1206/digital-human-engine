"""场景探测服务 - 根据 GPS 坐标推断用户当前场景"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GeoLocation:
    """地理位置"""
    latitude: float   # 纬度
    longitude: float  # 经度
    accuracy: float = 0.0  # 精度（米）


@dataclass
class SceneRegion:
    """场景区域"""
    scene_id: str
    name: str
    center: GeoLocation
    radius: float  # 覆盖半径（米）
    metadata: dict[str, Any] = None


class SceneDetector:
    """场景探测服务

    功能：
    1. 根据 GPS 坐标推断用户当前场景
    2. 支持多种定位源（GPS/Beacon/WiFi）
    3. 场景区域配置
    """

    # 预定义场景区域（生产环境应从数据库加载）
    DEFAULT_REGIONS: list[SceneRegion] = [
        # 医院（示例坐标，需替换为实际坐标）
        SceneRegion(
            scene_id="hospital",
            name="某医院",
            center=GeoLocation(latitude=31.2304, longitude=121.4737),
            radius=500,  # 500 米覆盖范围
            metadata={"type": "hospital", "address": "上海市黄浦区"}
        ),
        # 博物馆（示例坐标）
        SceneRegion(
            scene_id="museum",
            name="某博物馆",
            center=GeoLocation(latitude=31.2297, longitude=121.4692),
            radius=300,
            metadata={"type": "museum", "address": "上海市黄浦区"}
        ),
    ]

    def __init__(self) -> None:
        self._regions: list[SceneRegion] = self.DEFAULT_REGIONS.copy()
        self._cache: dict[str, tuple[str, float]] = {}  # 缓存：user_id -> (scene_id, timestamp)

    def add_region(self, region: SceneRegion) -> None:
        """添加场景区域"""
        self._regions.append(region)
        logger.info("场景区域添加: %s (%s)", region.scene_id, region.name)

    def set_regions(self, regions: list[SceneRegion]) -> None:
        """设置场景区域列表（替换默认）"""
        self._regions = regions
        logger.info("场景区域更新: 共 %d 个", len(regions))

    async def detect(
        self,
        location: GeoLocation,
        user_id: str = "",
        use_cache: bool = True,
        cache_ttl: float = 60.0,
    ) -> str | None:
        """探测用户当前场景

        Args:
            location: 用户位置
            user_id: 用户 ID（用于缓存）
            use_cache: 是否使用缓存
            cache_ttl: 缓存有效期（秒）

        Returns:
            scene_id，未匹配到返回 None
        """
        # 检查缓存
        if use_cache and user_id:
            cached = self._cache.get(user_id)
            if cached:
                scene_id, timestamp = cached
                if (self._now() - timestamp) < cache_ttl:
                    logger.debug("场景探测命中缓存: user=%s, scene=%s", user_id, scene_id)
                    return scene_id

        # 遍历所有区域，找到最近的匹配
        best_match: tuple[str, float] | None = None
        best_distance = float("inf")

        for region in self._regions:
            distance = self._calculate_distance(location, region.center)

            # 在覆盖范围内且距离最近
            if distance <= region.radius and distance < best_distance:
                best_distance = distance
                best_match = (region.scene_id, distance)

        if best_match:
            scene_id = best_match[0]
            # 更新缓存
            if user_id:
                self._cache[user_id] = (scene_id, self._now())

            logger.info(
                "场景探测成功: user=%s, scene=%s, distance=%.1fm",
                user_id, scene_id, best_distance
            )
            return scene_id

        logger.debug("场景探测未匹配: user=%s, lat=%.4f, lng=%.4f", user_id, location.latitude, location.longitude)
        return None

    async def detect_by_name(self, location_name: str) -> str | None:
        """根据位置名称推断场景（如 "消化内科" → "hospital"）

        Args:
            location_name: 位置名称（如科室名、展厅名）

        Returns:
            scene_id
        """
        # 简单关键词匹配（生产环境可用 NLP）
        hospital_keywords = ["科室", "门诊", "急诊", "住院", "挂号", "导诊", "药房", "检验"]
        museum_keywords = ["展厅", "展品", "文物", "藏品", "讲解", "导览", "博物馆"]

        for keyword in hospital_keywords:
            if keyword in location_name:
                return "hospital"

        for keyword in museum_keywords:
            if keyword in location_name:
                return "museum"

        return None

    def _calculate_distance(self, loc1: GeoLocation, loc2: GeoLocation) -> float:
        """计算两点间距离（米），使用 Haversine 公式"""
        R = 6371000  # 地球半径（米）

        lat1_rad = math.radians(loc1.latitude)
        lat2_rad = math.radians(loc2.latitude)
        delta_lat = math.radians(loc2.latitude - loc1.latitude)
        delta_lon = math.radians(loc2.longitude - loc1.longitude)

        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _now(self) -> float:
        """当前时间戳"""
        import time
        return time.time()

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("场景探测缓存已清空")


# 全局实例
scene_detector = SceneDetector()