"""
Решение уравнения линзы: поиск изображений для заданного положения источника.

- Аналитическое решение для точечной линзы
- Численное решение на сетке для произвольных моделей
"""

from __future__ import annotations

import numpy as np
from .models import LensModel, PointMass


def point_mass_images(beta_x: float, beta_y: float,
                      theta_e: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Аналитическое решение для точечной линзы в центре.

    β = θ - θ_E² / θ  →  θ = β/2 ± √(β²/4 + θ_E²)

    Возвращает (images_x, images_y) — массивы из двух изображений:
    [0] — «+» (дальнее, снаружи кольца Эйнштейна)
    [1] — «−» (ближнее, внутри кольца Эйнштейна)
    """
    beta = np.sqrt(beta_x**2 + beta_y**2)
    if beta < 1e-15:
        # Кольцо Эйнштейна — бесконечно много изображений на окружности θ_E
        # Возвращаем две «представительные» точки
        return np.array([theta_e, -theta_e]), np.array([0.0, 0.0])

    theta_plus = beta / 2 + np.sqrt((beta / 2)**2 + theta_e**2)
    theta_minus = beta / 2 - np.sqrt((beta / 2)**2 + theta_e**2)

    # Направление: вдоль вектора β
    cos_phi = beta_x / beta
    sin_phi = beta_y / beta

    images_x = np.array([theta_plus * cos_phi, theta_minus * cos_phi])
    images_y = np.array([theta_plus * sin_phi, theta_minus * sin_phi])
    return images_x, images_y


def find_images_grid(lens: LensModel, beta_x: float, beta_y: float,
                     grid_range: float = 3.0, grid_size: int = 500,
                     tol: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    Численный поиск изображений на сетке.

    Ищет точки θ, где |β(θ) - β_source| < tol.
    Затем группирует близкие точки и берёт центроид каждой группы.

    Returns
    -------
    images_x, images_y : массивы координат найденных изображений
    """
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x, y)

    bx, by = lens.lens_equation(X, Y)
    dist = np.sqrt((bx - beta_x)**2 + (by - beta_y)**2)

    mask = dist < tol
    if not mask.any():
        return np.array([]), np.array([])

    candidates_x = X[mask]
    candidates_y = Y[mask]

    # Группировка через connected components (label на маске)
    from scipy.ndimage import label
    labeled, n_features = label(mask)

    images_x_list = []
    images_y_list = []
    for i in range(1, n_features + 1):
        region = labeled == i
        # Взвешенный центроид (вес = 1/dist, ближе к точному решению — больше вес)
        d_region = dist[region]
        w = 1.0 / (d_region + 1e-15)
        images_x_list.append(np.average(X[region], weights=w))
        images_y_list.append(np.average(Y[region], weights=w))

    return np.array(images_x_list), np.array(images_y_list)


def critical_curves(lens: LensModel, grid_range: float = 3.0,
                    grid_size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Критические кривые: det(A) = 0, где A — матрица Якоби.

    det(A) = (1-κ)² - γ²  →  μ → ∞

    Возвращает координаты точек на критических кривых.
    """
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x, y)

    kappa = lens.convergence(X, Y)
    gamma = lens.shear(X, Y)
    det = (1 - kappa)**2 - gamma**2

    # Ищем смену знака — контурная линия det = 0
    from skimage.measure import find_contours
    contours = find_contours(det, 0.0)

    all_x, all_y = [], []
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    for contour in contours:
        all_x.append(x[0] + contour[:, 1] * dx)
        all_y.append(y[0] + contour[:, 0] * dy)

    if all_x:
        return np.concatenate(all_x), np.concatenate(all_y)
    return np.array([]), np.array([])


def caustics(lens: LensModel, grid_range: float = 3.0,
             grid_size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Каустики: образ критических кривых через уравнение линзы β = θ - α(θ).
    """
    cc_x, cc_y = critical_curves(lens, grid_range, grid_size)
    if len(cc_x) == 0:
        return np.array([]), np.array([])
    return lens.lens_equation(cc_x, cc_y)
