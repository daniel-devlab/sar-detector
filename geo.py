"""Pixel-to-ground helpers for telemetry-aware SAR detections."""

from __future__ import annotations

from dataclasses import dataclass
import math


EARTH_RADIUS_M = 6_378_137.0


@dataclass
class CameraPose:
    lat: float
    lon: float
    agl_m: float
    heading_deg: float
    gimbal_pitch_deg: float = 0.0
    gimbal_yaw_deg: float = 0.0
    hfov_deg: float = 70.0
    frame_w: int = 1920
    frame_h: int = 1080


@dataclass
class GroundPin:
    lat: float
    lon: float
    agl_m: float
    gsd_cm: float
    err_radius_m: float


def _meters_per_deg(lat_deg: float) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    meters_per_deg_lat = EARTH_RADIUS_M * math.pi / 180.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(lat)
    return meters_per_deg_lat, max(meters_per_deg_lon, 1e-9)


def offset_latlon(
    lat: float, lon: float, east_m: float, north_m: float
) -> tuple[float, float]:
    meters_per_deg_lat, meters_per_deg_lon = _meters_per_deg(lat)
    return lat + north_m / meters_per_deg_lat, lon + east_m / meters_per_deg_lon


def gsd_cm(pose: CameraPose) -> float:
    hfov = math.radians(pose.hfov_deg)
    width_m = 2.0 * pose.agl_m * math.tan(hfov / 2.0)
    return (width_m / max(pose.frame_w, 1)) * 100.0


def pixel_ray_cam(u: float, v: float, pose: CameraPose) -> tuple[float, float, float]:
    hfov = math.radians(pose.hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (pose.frame_h / max(pose.frame_w, 1)))
    x = (2.0 * (u + 0.5) / pose.frame_w - 1.0) * math.tan(hfov / 2.0)
    y = (2.0 * (v + 0.5) / pose.frame_h - 1.0) * math.tan(vfov / 2.0)
    z = 1.0
    norm = math.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def _rot_x(angle_rad: float) -> list[list[float]]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_z(angle_rad: float) -> list[list[float]]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _mul(
    matrix: list[list[float]], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def cam_to_ned(
    ray_cam: tuple[float, float, float], pose: CameraPose
) -> tuple[float, float, float]:
    rx, ry, rz = ray_cam
    body = (rx, ry, rz)
    pitch = math.radians(pose.gimbal_pitch_deg)
    yaw_gimbal = math.radians(pose.gimbal_yaw_deg)
    heading = math.radians(pose.heading_deg)

    body = _mul(_rot_x(-pitch), body)
    body = _mul(_rot_z(yaw_gimbal), body)

    east, south, down = body
    north = -south
    c, s = math.cos(heading), math.sin(heading)
    north_world = c * north - s * east
    east_world = s * north + c * east
    return north_world, east_world, down


def ground_hit(u: float, v: float, pose: CameraPose) -> GroundPin | None:
    if pose.agl_m <= 0:
        return None

    north, east, down = cam_to_ned(pixel_ray_cam(u, v, pose), pose)
    if down <= 1e-6:
        return None

    scale = pose.agl_m / down
    north_m = north * scale
    east_m = east * scale
    lat, lon = offset_latlon(pose.lat, pose.lon, east_m, north_m)
    cm = gsd_cm(pose)
    err = (2.0 * cm / 100.0) + 0.10 * pose.agl_m * math.tan(
        math.radians(pose.hfov_deg) / 2.0
    ) * 0.15
    return GroundPin(
        lat=lat,
        lon=lon,
        agl_m=pose.agl_m,
        gsd_cm=cm,
        err_radius_m=max(err, cm / 100.0 * 3.0),
    )


def box_center_pin(xyxy, pose: CameraPose) -> GroundPin | None:
    x_min, y_min, x_max, y_max = (float(v) for v in xyxy)
    return ground_hit((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, pose)
