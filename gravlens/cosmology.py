"""
Космологические расстояния и параметры линзирования.

Основано на формулах из Cosmology(4).ipynb.
Плоская ΛCDM-космология.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate


# Константы
C_KM_S = 299792.458  # скорость света, км/с
G_SI = 6.674e-11      # гравитационная постоянная, м³/(кг·с²)
M_SUN = 1.989e30       # масса Солнца, кг
MPC_TO_M = 3.0857e22   # мегапарсек в метрах
ARCSEC_TO_RAD = np.pi / (180 * 3600)


class Cosmology:
    """Плоская ΛCDM-космология."""

    def __init__(self, H0: float = 70.0, omega_m: float = 0.3):
        """
        Parameters
        ----------
        H0 : float
            Постоянная Хаббла, км/с/Мпк
        omega_m : float
            Параметр плотности материи Ω_m. Ω_Λ = 1 - Ω_m.
        """
        self.H0 = H0
        self.omega_m = omega_m
        self.omega_L = 1.0 - omega_m

    def _integrand(self, z):
        return 1.0 / np.sqrt(self.omega_m * (1 + z)**3 + self.omega_L)

    def comoving_distance(self, z: float) -> float:
        """Сопутствующее расстояние d_C(z) в Мпк."""
        result, _ = integrate.quad(self._integrand, 0, z)
        return result * C_KM_S / self.H0

    def angular_diameter_distance(self, z1: float, z2: float | None = None) -> float:
        """
        Угловое расстояние D_A в Мпк.

        - D_A(z) если z2 не задан (от наблюдателя до z1)
        - D_A(z1, z2) если z2 задан (от z1 до z2, z2 > z1)
        """
        if z2 is None:
            z2 = z1
            z1 = 0.0
        result, _ = integrate.quad(self._integrand, z1, z2)
        return result * C_KM_S / self.H0 / (1 + z2)

    def luminosity_distance(self, z: float) -> float:
        """Расстояние светимости D_L(z) в Мпк."""
        return self.angular_diameter_distance(z) * (1 + z)**2

    def critical_density(self, z_lens: float, z_source: float) -> float:
        """
        Критическая поверхностная плотность Σ_cr в кг/м².

        Σ_cr = c² / (4π G) · D_s / (D_l · D_ls)
        """
        D_l = self.angular_diameter_distance(z_lens) * MPC_TO_M
        D_s = self.angular_diameter_distance(z_source) * MPC_TO_M
        D_ls = self.angular_diameter_distance(z_lens, z_source) * MPC_TO_M
        c_si = C_KM_S * 1e3  # м/с
        return c_si**2 / (4 * np.pi * G_SI) * D_s / (D_l * D_ls)

    def einstein_radius(self, mass_kg: float, z_lens: float, z_source: float) -> float:
        """
        Радиус Эйнштейна θ_E в угловых секундах.

        θ_E = √(4GM/c² · D_ls / (D_l · D_s))
        """
        D_l = self.angular_diameter_distance(z_lens) * MPC_TO_M
        D_s = self.angular_diameter_distance(z_source) * MPC_TO_M
        D_ls = self.angular_diameter_distance(z_lens, z_source) * MPC_TO_M
        c_si = C_KM_S * 1e3
        theta_rad = np.sqrt(4 * G_SI * mass_kg / c_si**2 * D_ls / (D_l * D_s))
        return theta_rad / ARCSEC_TO_RAD

    def einstein_radius_solar(self, mass_solar: float, z_lens: float, z_source: float) -> float:
        """Радиус Эйнштейна (масса в массах Солнца) в угловых секундах."""
        return self.einstein_radius(mass_solar * M_SUN, z_lens, z_source)

    def mass_from_einstein_radius(self, theta_e_arcsec: float,
                                   z_lens: float, z_source: float) -> float:
        """
        Масса линзы по радиусу Эйнштейна.

        Возвращает массу в массах Солнца.
        M = θ_E² c² D_l D_s / (4G D_ls)
        """
        D_l = self.angular_diameter_distance(z_lens) * MPC_TO_M
        D_s = self.angular_diameter_distance(z_source) * MPC_TO_M
        D_ls = self.angular_diameter_distance(z_lens, z_source) * MPC_TO_M
        c_si = C_KM_S * 1e3
        theta_rad = theta_e_arcsec * ARCSEC_TO_RAD
        mass_kg = theta_rad**2 * c_si**2 / (4 * G_SI) * D_l * D_s / D_ls
        return mass_kg / M_SUN
