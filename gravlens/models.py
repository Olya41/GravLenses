"""
Модели гравитационных линз.

Каждая модель задаёт:
- potential(theta_x, theta_y)   — линзирующий потенциал ψ(θ)
- deflection(theta_x, theta_y)  — угол отклонения α(θ) = ∇ψ
- convergence(theta_x, theta_y) — поверхностная плотность κ(θ) = ½ ∇²ψ
- shear(theta_x, theta_y)       — сдвиг γ(θ)

Все величины безразмерные, нормированы на радиус Эйнштейна θ_E.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class LensModel(ABC):
    """Базовый класс модели линзы."""

    @abstractmethod
    def potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Линзирующий потенциал ψ(θ)."""

    @abstractmethod
    def deflection(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Угол отклонения (α_x, α_y) = ∇ψ."""

    @abstractmethod
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Конвергенция κ = Σ / Σ_cr."""

    def shear(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Модуль сдвига |γ|. По умолчанию — численно из потенциала."""
        h = 1e-5
        psi = self.potential
        psi_xx = (psi(x + h, y) - 2 * psi(x, y) + psi(x - h, y)) / h**2
        psi_yy = (psi(x, y + h) - 2 * psi(x, y) + psi(x, y - h)) / h**2
        psi_xy = (psi(x + h, y + h) - psi(x + h, y - h) - psi(x - h, y + h) + psi(x - h, y - h)) / (4 * h**2)
        gamma1 = 0.5 * (psi_xx - psi_yy)
        gamma2 = psi_xy
        return np.sqrt(gamma1**2 + gamma2**2)

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Усиление μ = 1 / ((1-κ)² - γ²)."""
        kappa = self.convergence(x, y)
        gamma = self.shear(x, y)
        det = (1 - kappa)**2 - gamma**2
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 1.0 / det
        return mu

    def lens_equation(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """β = θ - α(θ). Возвращает положение источника по положению изображения."""
        ax, ay = self.deflection(x, y)
        return x - ax, y - ay


class PointMass(LensModel):
    """
    Точечная масса (линза Шварцшильда).

    ψ(θ) = ln|θ|
    α(θ) = θ / |θ|²
    κ = 0 (кроме центра)
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, einstein_radius: float = 1.0):
        self.x0 = x0
        self.y0 = y0
        self.theta_e = einstein_radius

    def _r2(self, x, y):
        return (x - self.x0)**2 + (y - self.y0)**2

    def potential(self, x, y):
        r2 = self._r2(x, y)
        return self.theta_e**2 * np.log(np.sqrt(r2))

    def deflection(self, x, y):
        r2 = self._r2(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax = self.theta_e**2 * (x - self.x0) / r2
            ay = self.theta_e**2 * (y - self.y0) / r2
        return ax, ay

    def convergence(self, x, y):
        return np.zeros_like(x, dtype=float)

    def shear(self, x, y):
        r2 = self._r2(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.theta_e**2 / r2

    def magnification(self, x, y):
        r2 = self._r2(x, y)
        r4 = r2**2
        te4 = self.theta_e**4
        with np.errstate(divide="ignore", invalid="ignore"):
            return r4 / (r4 - te4)


class SIS(LensModel):
    """
    Сингулярная изотермическая сфера.

    ψ(θ) = θ_E |θ|
    α(θ) = θ_E θ/|θ|
    κ = θ_E / (2|θ|)
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, einstein_radius: float = 1.0):
        self.x0 = x0
        self.y0 = y0
        self.theta_e = einstein_radius

    def _r(self, x, y):
        return np.sqrt((x - self.x0)**2 + (y - self.y0)**2)

    def potential(self, x, y):
        return self.theta_e * self._r(x, y)

    def deflection(self, x, y):
        r = self._r(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax = self.theta_e * (x - self.x0) / r
            ay = self.theta_e * (y - self.y0) / r
        return ax, ay

    def convergence(self, x, y):
        r = self._r(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.theta_e / (2 * r)

    def shear(self, x, y):
        return self.convergence(x, y)


class NFW(LensModel):
    """
    Профиль Наварро—Френка—Уайта (NFW).

    Аппроксимация для гало тёмной материи.
    κ_s и r_s — характерная конвергенция и масштабный радиус.
    """

    def __init__(self, kappa_s: float = 0.2, r_s: float = 1.0,
                 x0: float = 0.0, y0: float = 0.0):
        self.kappa_s = kappa_s
        self.r_s = r_s
        self.x0 = x0
        self.y0 = y0

    def _xi(self, x, y):
        return np.sqrt((x - self.x0)**2 + (y - self.y0)**2) / self.r_s

    @staticmethod
    def _g(xi):
        """Функция g(ξ) для NFW-профиля."""
        result = np.zeros_like(xi, dtype=float)
        lt = xi < 1
        gt = xi > 1
        eq = np.isclose(xi, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[lt] = np.log(xi[lt] / 2) + 1 / np.sqrt(1 - xi[lt]**2) * np.arccosh(1 / xi[lt])
            result[gt] = np.log(xi[gt] / 2) + 1 / np.sqrt(xi[gt]**2 - 1) * np.arccos(1 / xi[gt])
            result[eq] = 1 + np.log(0.5)
        return result

    @staticmethod
    def _h(xi):
        """Функция h(ξ) для конвергенции NFW."""
        result = np.zeros_like(xi, dtype=float)
        lt = xi < 1
        gt = xi > 1
        eq = np.isclose(xi, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[lt] = 1 / (xi[lt]**2 - 1) * (1 - 1 / np.sqrt(1 - xi[lt]**2) * np.arccosh(1 / xi[lt]))
            result[gt] = 1 / (xi[gt]**2 - 1) * (1 - 1 / np.sqrt(xi[gt]**2 - 1) * np.arccos(1 / xi[gt]))
            result[eq] = 1.0 / 3.0
        return result

    def potential(self, x, y):
        xi = self._xi(x, y)
        return 2 * self.kappa_s * self.r_s**2 * self._g(xi)

    def deflection(self, x, y):
        xi = self._xi(x, y)
        r2 = (x - self.x0)**2 + (y - self.y0)**2
        g = self._g(xi)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = 2 * self.kappa_s * self.r_s**2 * g / r2
        return factor * (x - self.x0), factor * (y - self.y0)

    def convergence(self, x, y):
        xi = self._xi(x, y)
        return 2 * self.kappa_s * self._h(xi)


class CompositeLens(LensModel):
    """Композитная линза: сумма нескольких моделей."""

    def __init__(self, *lenses: LensModel):
        self.lenses = list(lenses)

    def add(self, lens: LensModel):
        self.lenses.append(lens)

    def potential(self, x, y):
        return sum(lens.potential(x, y) for lens in self.lenses)

    def deflection(self, x, y):
        ax_total = np.zeros_like(x, dtype=float)
        ay_total = np.zeros_like(y, dtype=float)
        for lens in self.lenses:
            ax, ay = lens.deflection(x, y)
            ax_total += ax
            ay_total += ay
        return ax_total, ay_total

    def convergence(self, x, y):
        return sum(lens.convergence(x, y) for lens in self.lenses)
