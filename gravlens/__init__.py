"""
gravlens — библиотека гравитационного линзирования.

Модули:
    models         — модели линз (PointMass, SIS, NFW, CompositeLens)
    cosmology      — космологические расстояния и радиус Эйнштейна
    solvers        — решение уравнения линзы, критические кривые, каустики
    magnification  — усиление, площади, формула Погсона
    plotting       — тепловые карты, эквипотенциали, карты усиления
    interactive    — интерактивная визуализация
"""

from .models import PointMass, SIS, NFW, CompositeLens, LensModel
from .cosmology import Cosmology
from .solvers import point_mass_images, find_images_grid, critical_curves, caustics
from .magnification import (
    shoelace_area, pogson_delta_m, flux_ratio, total_magnification,
    generate_ellipse_boundary, generate_ellipse_fill, is_inside_ellipse,
)
from .plotting import (
    plot_potential, plot_convergence, plot_magnification_map,
    plot_critical_and_caustic, plot_deflection_field,
)
from .interactive import LensingExplorer
