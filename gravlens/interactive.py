"""
Интерактивная визуализация гравитационного линзирования.

Перетаскивание источника мышкой, слайдеры параметров,
визуализация изображений, площадей и усилений.

Рефакторинг Code.py / itog.py с использованием библиотеки gravlens.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Slider, CheckButtons

from .models import PointMass
from .solvers import point_mass_images
from .magnification import (
    shoelace_area,
    pogson_delta_m,
    generate_ellipse_boundary,
    is_inside_ellipse,
)


class LensingExplorer:
    """
    Интерактивная визуализация точечной гравитационной линзы.

    Использование:
        explorer = LensingExplorer()
        explorer.show()
    """

    def __init__(self, theta_e: float = 1.0, grid_lim: float = 3.0):
        self.lens = PointMass(einstein_radius=theta_e)
        self.theta_e = theta_e
        self.grid_lim = grid_lim

        # Параметры эллипса источника
        self.semi_a = 0.25
        self.semi_b = 0.20
        self.n_boundary = 500
        self.angle_deg = 0.0

        # Видимость
        self.ring_visible = True
        self.boundary_visible = True
        self.images_visible = True

        self._build_figure()
        self._build_sliders()
        self._build_checkboxes()
        self._connect_events()

        self.dragging = False
        self.update_all(1.0, 0.0)

    def _build_figure(self):
        # Лейаут:
        #   Верхняя часть: [график 65%] [инфо-панель 30%]
        #   Нижняя часть: слайдеры + чекбоксы
        self.fig = plt.figure(figsize=(20, 13), dpi=100)

        # Основной график — квадратный, слева, с отступом снизу под слайдеры
        self.ax = self.fig.add_axes([0.06, 0.18, 0.55, 0.75])
        self.ax.set_xlim(-self.grid_lim, self.grid_lim)
        self.ax.set_ylim(-self.grid_lim, self.grid_lim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Гравитационное линзирование", fontsize=16, pad=10)

        # Информационная панель справа (невидимые оси для текста)
        self.info_ax = self.fig.add_axes([0.65, 0.18, 0.33, 0.75])
        self.info_ax.set_axis_off()
        self.info_text = self.info_ax.text(
            0.05, 0.95, "", transform=self.info_ax.transAxes,
            fontsize=13, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
        )

        # Кольцо Эйнштейна
        self.einstein_circle = Circle(
            (0, 0), self.theta_e, fill=False,
            color="blue", linewidth=2, linestyle="--", alpha=0.7,
        )
        self.ax.add_patch(self.einstein_circle)

        # Точки: источник, два изображения
        self.source_dot = Circle((1, 0), 0.03, color="red", alpha=0.8, zorder=10)
        self.img1_dot = Circle((0, 0), 0.03, color="green", alpha=0.8, zorder=10)
        self.img2_dot = Circle((0, 0), 0.03, color="blue", alpha=0.8, zorder=10)
        for p in (self.source_dot, self.img1_dot, self.img2_dot):
            self.ax.add_patch(p)

        # Эллипс источника
        self.source_ellipse = Ellipse(
            (1, 0), width=2 * self.semi_a, height=2 * self.semi_b,
            fill=False, color="red", linewidth=1.5, alpha=0.5,
        )
        self.ax.add_patch(self.source_ellipse)

        # Scatter для точек границ
        self.scatter_src = None
        self.scatter_img1 = None
        self.scatter_img2 = None

        # Линии от источника к изображениям
        self.line1, = self.ax.plot([], [], "--", color="gray", alpha=0.2)
        self.line2, = self.ax.plot([], [], "--", color="gray", alpha=0.2)

    def _build_sliders(self):
        # Слайдеры — горизонтальная полоса внизу, под графиком
        sl = 0.15          # left
        sw = 0.45          # width
        sh = 0.018         # height
        gap = 0.030        # расстояние между слайдерами

        base_y = 0.03      # нижний слайдер
        self.slider_n = Slider(
            self.fig.add_axes([sl, base_y, sw, sh]),
            "Точки", 100, 5000, valinit=self.n_boundary, valstep=50)
        self.slider_b = Slider(
            self.fig.add_axes([sl, base_y + gap, sw, sh]),
            "Полуось B", 0.05, 0.5, valinit=self.semi_b)
        self.slider_a = Slider(
            self.fig.add_axes([sl, base_y + 2 * gap, sw, sh]),
            "Полуось A", 0.05, 0.5, valinit=self.semi_a)
        self.slider_angle = Slider(
            self.fig.add_axes([sl, base_y + 3 * gap, sw, sh]),
            "Угол (°)", -90, 90, valinit=0)

        for s in (self.slider_angle, self.slider_a, self.slider_b, self.slider_n):
            s.label.set_fontsize(11)
            s.valtext.set_fontsize(11)

        self.slider_angle.on_changed(lambda v: self._on_param_change(angle=v))
        self.slider_a.on_changed(lambda v: self._on_param_change(a=v))
        self.slider_b.on_changed(lambda v: self._on_param_change(b=v))
        self.slider_n.on_changed(lambda v: self._on_param_change(n=int(v)))

    def _build_checkboxes(self):
        # Чекбоксы — справа от слайдеров, внизу
        cl = 0.68
        cw = 0.12
        ch = 0.10

        ax_cb = self.fig.add_axes([cl, 0.02, cw, ch])
        self.cb_all = CheckButtons(ax_cb, ["Кольцо", "Источник", "Изобр."], [True, True, True])
        for label in self.cb_all.labels:
            label.set_fontsize(11)
        self.cb_all.on_clicked(self._on_checkbox_clicked)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # --- Обработчики событий ---

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.source_dot.contains(event)
        if contains:
            self.dragging = True

    def _on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        self.update_all(event.xdata, event.ydata)
        self.fig.canvas.draw()

    def _on_release(self, event):
        self.dragging = False

    def _on_param_change(self, angle=None, a=None, b=None, n=None):
        if angle is not None:
            self.angle_deg = angle
        if a is not None:
            self.semi_a = a
            self.source_ellipse.width = 2 * a
        if b is not None:
            self.semi_b = b
            self.source_ellipse.height = 2 * b
        if n is not None:
            self.n_boundary = n
        x, y = self.source_dot.center
        self.update_all(x, y)
        self.fig.canvas.draw_idle()

    def _on_checkbox_clicked(self, label):
        if label == "Кольцо":
            self.ring_visible = not self.ring_visible
            self.einstein_circle.set_visible(self.ring_visible)
        elif label == "Источник":
            self.boundary_visible = not self.boundary_visible
            if self.scatter_src is not None:
                self.scatter_src.set_visible(self.boundary_visible)
        elif label == "Изобр.":
            self.images_visible = not self.images_visible
            for sc in (self.scatter_img1, self.scatter_img2):
                if sc is not None:
                    sc.set_visible(self.images_visible)
        self.fig.canvas.draw_idle()

    # --- Основная логика ---

    def update_all(self, x: float, y: float):
        """Обновляет всю визуализацию для положения источника (x, y)."""
        self.source_dot.center = (x, y)
        self.source_ellipse.center = (x, y)
        self.source_ellipse.angle = self.angle_deg

        # Изображения
        imgs_x, imgs_y = point_mass_images(x, y, self.theta_e)
        self.img1_dot.center = (imgs_x[0], imgs_y[0])
        self.img2_dot.center = (imgs_x[1], imgs_y[1])

        # Линии
        self.line1.set_data([x, imgs_x[0]], [y, imgs_y[0]])
        self.line2.set_data([x, imgs_x[1]], [y, imgs_y[1]])

        # Точки на границе
        self._update_boundary_points(x, y, imgs_x, imgs_y)

        # Информационная панель
        self._update_info(x, y, imgs_x, imgs_y)

    def _update_boundary_points(self, x, y, imgs_x, imgs_y):
        for sc in (self.scatter_src, self.scatter_img1, self.scatter_img2):
            if sc is not None:
                sc.remove()
        self.scatter_src = self.scatter_img1 = self.scatter_img2 = None

        bx, by = generate_ellipse_boundary(
            x, y, self.semi_a, self.semi_b, self.n_boundary, self.angle_deg)

        self.scatter_src = self.ax.scatter(bx, by, c="red", s=5, alpha=0.3, zorder=1)
        self.scatter_src.set_visible(self.boundary_visible)

        # Сохраняем для площади
        self._src_bx, self._src_by = bx, by

        # Трансформация через формулу линзы
        r = np.sqrt(bx**2 + by**2)
        valid = r > 1e-10

        bx_v, by_v, r_v = bx[valid], by[valid], r[valid]
        theta_p = r_v / 2 + np.sqrt((r_v / 2)**2 + self.theta_e**2)
        theta_m = r_v / 2 - np.sqrt((r_v / 2)**2 + self.theta_e**2)

        f1x = theta_p * bx_v / r_v
        f1y = theta_p * by_v / r_v
        f2x = theta_m * bx_v / r_v
        f2y = theta_m * by_v / r_v

        self.scatter_img1 = self.ax.scatter(f1x, f1y, c="green", s=5, alpha=0.3, zorder=1)
        self.scatter_img2 = self.ax.scatter(f2x, f2y, c="blue", s=5, alpha=0.3, zorder=1)
        self.scatter_img1.set_visible(self.images_visible)
        self.scatter_img2.set_visible(self.images_visible)

        self._f1x, self._f1y = f1x, f1y
        self._f2x, self._f2y = f2x, f2y

    def _update_info(self, x, y, imgs_x, imgs_y):
        area_src = shoelace_area(self._src_bx, self._src_by)

        origin_inside = is_inside_ellipse(0, 0, x, y, self.semi_a, self.semi_b, self.angle_deg)
        einstein_area = np.pi * self.theta_e**2

        area_f1_raw = shoelace_area(self._f1x, self._f1y) if len(self._f1x) >= 3 else 0.0
        area_f2_raw = shoelace_area(self._f2x, self._f2y) if len(self._f2x) >= 3 else 0.0

        if origin_inside:
            area_f1 = area_f1_raw - einstein_area
            area_f2 = einstein_area - area_f2_raw
        else:
            area_f1 = area_f1_raw
            area_f2 = area_f2_raw

        if area_src > 0:
            k1 = area_f1 / area_src
            k2 = area_f2 / area_src
            dm1 = pogson_delta_m(k1) if k1 > 0 else 0.0
            dm2 = pogson_delta_m(k2) if k2 > 0 else 0.0
        else:
            k1 = k2 = dm1 = dm2 = 0.0

        info = (
            "Координаты центров\n"
            "─────────────────────────\n"
            f"  Источник     ({x:+.4f}, {y:+.4f})\n"
            f"  Изобр. 1     ({imgs_x[0]:+.4f}, {imgs_y[0]:+.4f})\n"
            f"  Изобр. 2     ({imgs_x[1]:+.4f}, {imgs_y[1]:+.4f})\n"
            "\n"
            "Площади\n"
            "─────────────────────────\n"
            f"  S источника  {area_src:.5f}\n"
            f"  S изобр. 1   {area_f1:.5f}\n"
            f"  S изобр. 2   {area_f2:.5f}\n"
            "\n"
            "Усиление\n"
            "─────────────────────────\n"
            f"  k₁ = {k1:.4f}    k₂ = {k2:.4f}\n"
            f"  Δm₁ = {dm1:+.2f}   Δm₂ = {dm2:+.2f}\n"
            "\n"
            "Обозначения\n"
            "─────────────────────────\n"
            "  ● Источник (красный)\n"
            "  ● Изобр. 1 (зелёный)\n"
            "  ● Изобр. 2 (синий)\n"
            "  - - Кольцо Эйнштейна"
        )

        self.info_text.set_text(info)

    def show(self):
        plt.show()
