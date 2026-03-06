"""
Визуализация: тепловые карты потенциалов, эквипотенциали,
критические кривые, каустики, карты усиления.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .models import LensModel
from .solvers import critical_curves, caustics


def make_grid(grid_range: float = 2.0, grid_size: int = 500):
    """Создаёт 2D-сетку."""
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    return np.meshgrid(x, y)


def plot_potential(lens: LensModel, grid_range: float = 2.0, grid_size: int = 500,
                   mask_radius: float = 0.01, n_contours: int = 25,
                   cmap: str = "inferno", ax=None, colorbar: bool = True,
                   show_contours: bool = True):
    """
    Тепловая карта потенциала с эквипотенциалями.

    Parameters
    ----------
    lens : модель линзы
    grid_range : половина стороны области [-R, R]
    mask_radius : радиус выколотой окрестности вокруг сингулярностей
    n_contours : число линий эквипотенциалей
    """
    X, Y = make_grid(grid_range, grid_size)
    psi = lens.potential(X, Y)

    # Маскируем сингулярности (конвергенция → ∞)
    kappa = lens.convergence(X, Y)
    mask = ~np.isfinite(psi) | ~np.isfinite(kappa)
    # Маскируем окрестности центров (для составных линз — по позициям)
    if hasattr(lens, "lenses"):
        for sub in lens.lenses:
            x0 = getattr(sub, "x0", 0)
            y0 = getattr(sub, "y0", 0)
            r = np.sqrt((X - x0)**2 + (Y - y0)**2)
            mask |= r < mask_radius
    else:
        x0 = getattr(lens, "x0", 0)
        y0 = getattr(lens, "y0", 0)
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        mask |= r < mask_radius

    psi_masked = np.ma.array(psi, mask=mask)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    im = ax.pcolormesh(X, Y, psi_masked, shading="auto", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_x$")
    ax.set_ylabel(r"$\theta_y$")

    if show_contours and psi_masked.count() > 0:
        levels = np.linspace(psi_masked.min(), psi_masked.max(), n_contours)
        ax.contour(X, Y, psi_masked, levels=levels, colors="white", linewidths=0.6, alpha=0.7)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, location="left", pad=0.12)
        cbar.set_label(r"$\psi(\theta)$")

    return fig, ax


def plot_convergence(lens: LensModel, grid_range: float = 2.0, grid_size: int = 500,
                     cmap: str = "magma", ax=None, colorbar: bool = True):
    """Карта конвергенции κ(θ)."""
    X, Y = make_grid(grid_range, grid_size)
    kappa = lens.convergence(X, Y)
    kappa_masked = np.ma.array(kappa, mask=~np.isfinite(kappa))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    im = ax.pcolormesh(X, Y, kappa_masked, shading="auto", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_x$")
    ax.set_ylabel(r"$\theta_y$")
    ax.set_title(r"$\kappa(\theta)$")

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\kappa$")

    return fig, ax


def plot_magnification_map(lens: LensModel, grid_range: float = 2.0, grid_size: int = 500,
                            cmap: str = "RdBu_r", ax=None, colorbar: bool = True,
                            vmax: float = 10.0):
    """Карта усиления μ(θ) = 1/det(A)."""
    X, Y = make_grid(grid_range, grid_size)
    mu = lens.magnification(X, Y)
    mu_clipped = np.clip(mu, -vmax, vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    im = ax.pcolormesh(X, Y, mu_clipped, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_x$")
    ax.set_ylabel(r"$\theta_y$")
    ax.set_title(r"$\mu(\theta)$")

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\mu$")

    return fig, ax


def plot_critical_and_caustic(lens: LensModel, grid_range: float = 3.0,
                               grid_size: int = 500, ax=None):
    """Критические кривые и каустики на одном графике."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    try:
        cc_x, cc_y = critical_curves(lens, grid_range, grid_size)
        ca_x, ca_y = caustics(lens, grid_range, grid_size)

        if len(cc_x) > 0:
            ax.plot(cc_x, cc_y, "r.", markersize=0.5, label="Критические кривые")
        if len(ca_x) > 0:
            ax.plot(ca_x, ca_y, "b.", markersize=0.5, label="Каустики")
    except ImportError:
        ax.text(0.5, 0.5, "Нужен scikit-image\npip install scikit-image",
                transform=ax.transAxes, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_x$")
    ax.set_ylabel(r"$\theta_y$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_deflection_field(lens: LensModel, grid_range: float = 2.0,
                           grid_size: int = 30, ax=None):
    """Поле углов отклонения (quiver plot)."""
    X, Y = make_grid(grid_range, grid_size)
    ax_x, ax_y = lens.deflection(X, Y)

    if ax is None:
        fig, ax_plot = plt.subplots(figsize=(7, 7))
    else:
        ax_plot = ax
        fig = ax.figure

    ax_plot.quiver(X, Y, ax_x, ax_y, np.sqrt(ax_x**2 + ax_y**2), cmap="viridis")
    ax_plot.set_aspect("equal")
    ax_plot.set_xlabel(r"$\theta_x$")
    ax_plot.set_ylabel(r"$\theta_y$")
    ax_plot.set_title(r"$\alpha(\theta)$")

    return fig, ax_plot
