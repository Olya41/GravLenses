"""
Вычисление усиления, площадей и звёздных величин.

Формулы:
- μ = 1 / det(A) = r⁴/(r⁴ - θ_E⁴) для точечной линзы
- Δm = -2.5 lg(μ) — формула Погсона
- Площадь по формуле Гаусса (shoelace)
"""

from __future__ import annotations

import numpy as np


def shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Площадь замкнутого многоугольника по формуле Гаусса.

    Точки должны быть упорядочены по контуру.
    S = ½ |Σ (x_i · y_{i+1} − x_{i+1} · y_i)|
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def pogson_delta_m(mu: float | np.ndarray) -> float | np.ndarray:
    """
    Разность звёздных величин: Δm = -2.5 lg(|μ|).

    Δm < 0 означает усиление (объект ярче).
    """
    mu = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = -2.5 * np.log10(np.abs(mu))
    return float(result) if result.ndim == 0 else result


def flux_ratio(mu1: float, mu2: float) -> float:
    """Отношение потоков двух изображений: |μ1/μ2|."""
    return abs(mu1 / mu2)


def total_magnification(magnifications: np.ndarray) -> float:
    """Полное усиление: сумма |μ_i| по всем изображениям."""
    return float(np.sum(np.abs(magnifications)))


def generate_ellipse_boundary(cx: float, cy: float, a: float, b: float,
                               n_points: int = 500,
                               angle_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерирует точки на границе эллипса.

    Parameters
    ----------
    cx, cy : центр
    a, b : полуоси
    n_points : число точек
    angle_deg : угол поворота (градусы)
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    ang = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    return cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a


def generate_ellipse_fill(cx: float, cy: float, a: float, b: float,
                           n_points: int = 4000,
                           angle_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерирует точки внутри эллипса (равномерно по площади).
    """
    n_radial = max(10, int(np.sqrt(n_points)))
    n_theta = max(12, n_points // n_radial)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radii = np.sqrt(np.linspace(0, 1, n_radial, endpoint=False))
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    x = a * rr.ravel() * np.cos(tt.ravel())
    y = b * rr.ravel() * np.sin(tt.ravel())
    ang = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    return cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a


def is_inside_ellipse(px, py, cx, cy, a, b, angle_deg=0.0):
    """Проверяет, лежит ли точка внутри эллипса."""
    ang = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    dx, dy = px - cx, py - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    return (xr / a)**2 + (yr / b)**2 < 1.0
