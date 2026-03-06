"""
Интерактивное векторное поле градиента потенциала двух линз.

Переключение: точечные массы / SIS.
Слайдеры: theta_E1, theta_E2, расстояние.
Круги Эйнштейна + стрелки градиентов (= углы отклонения).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


class DualLensGradientExplorer:

    def __init__(self):
        self.te1 = 1.0
        self.te2 = 1.0
        self.sep = 0.3
        self.model = "Point Mass"

        self.grid_range = 2.5
        self.N_quiver = 30  # сетка для стрелок (реже — читаемее)

        self._build_figure()
        self._build_sliders()
        self._draw()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 11), dpi=100)
        self.ax = self.fig.add_axes([0.08, 0.22, 0.82, 0.72])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel(r"$\theta_x$", fontsize=13)
        self.ax.set_ylabel(r"$\theta_y$", fontsize=13)

    def _build_sliders(self):
        sl, sw, sh = 0.15, 0.50, 0.02
        gap = 0.035
        base = 0.02

        self.slider_te1 = Slider(
            self.fig.add_axes([sl, base + 2 * gap, sw, sh]),
            r"$\theta_{E1}$", 0.1, 3.0, valinit=self.te1)
        self.slider_te2 = Slider(
            self.fig.add_axes([sl, base + gap, sw, sh]),
            r"$\theta_{E2}$", 0.1, 3.0, valinit=self.te2)
        self.slider_sep = Slider(
            self.fig.add_axes([sl, base, sw, sh]),
            "Расстояние", 0.0, 4.0, valinit=self.sep)

        for s in (self.slider_te1, self.slider_te2, self.slider_sep):
            s.label.set_fontsize(12)
            s.valtext.set_fontsize(12)
            s.on_changed(lambda _: self._on_change())

        ax_radio = self.fig.add_axes([0.72, 0.02, 0.14, 0.08])
        self.radio = RadioButtons(ax_radio, ["Point Mass", "SIS"], active=0)
        for label in self.radio.labels:
            label.set_fontsize(11)
        self.radio.on_clicked(self._on_model_change)

    def _on_change(self):
        self.te1 = self.slider_te1.val
        self.te2 = self.slider_te2.val
        self.sep = self.slider_sep.val
        self._draw()
        self.fig.canvas.draw_idle()

    def _on_model_change(self, label):
        self.model = label
        self._draw()
        self.fig.canvas.draw_idle()

    def _gradient_point_mass(self, X, Y, x0, te):
        """grad(psi) = te^2 / r^2 * (dx, dy), где dx = X - x0"""
        dx = X - x0
        dy = Y
        r2 = dx**2 + dy**2
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = te**2 / r2
        return factor * dx, factor * dy

    def _gradient_sis(self, X, Y, x0, te):
        """grad(psi) = te / r * (dx, dy)"""
        dx = X - x0
        dy = Y
        r = np.sqrt(dx**2 + dy**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = te / r
        return factor * dx, factor * dy

    def _draw(self):
        d = self.sep / 2
        x1, x2 = -d, d

        # Сетка для стрелок
        x = np.linspace(-self.grid_range, self.grid_range, self.N_quiver)
        y = np.linspace(-self.grid_range, self.grid_range, self.N_quiver)
        X, Y = np.meshgrid(x, y)

        if self.model == "Point Mass":
            gx1, gy1 = self._gradient_point_mass(X, Y, x1, self.te1)
            gx2, gy2 = self._gradient_point_mass(X, Y, x2, self.te2)
            title = (r"Point Mass:  $\alpha = \nabla\psi = "
                     r"\theta_{E1}^2\,\hat{r}_1/r_1 + \theta_{E2}^2\,\hat{r}_2/r_2$")
        else:
            gx1, gy1 = self._gradient_sis(X, Y, x1, self.te1)
            gx2, gy2 = self._gradient_sis(X, Y, x2, self.te2)
            title = (r"SIS:  $\alpha = \nabla\psi = "
                     r"\theta_{E1}\,\hat{r}_1 + \theta_{E2}\,\hat{r}_2$")

        Gx = gx1 + gx2
        Gy = gy1 + gy2

        # Маскировка вблизи центров линз
        r1 = np.sqrt((X - x1)**2 + Y**2)
        r2 = np.sqrt((X - x2)**2 + Y**2)
        mask = (r1 < 0.15) | (r2 < 0.15)
        Gx[mask] = np.nan
        Gy[mask] = np.nan

        # Длина для окраски
        mag = np.sqrt(Gx**2 + Gy**2)

        # --- Перерисовка ---
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel(r"$\theta_x$", fontsize=13)
        self.ax.set_ylabel(r"$\theta_y$", fontsize=13)
        self.ax.set_title(title, fontsize=13)
        self.ax.set_xlim(-self.grid_range, self.grid_range)
        self.ax.set_ylim(-self.grid_range, self.grid_range)
        self.ax.set_facecolor("#1a1a2e")

        # Векторное поле
        self.ax.quiver(
            X, Y, Gx, Gy, mag,
            cmap="plasma", scale=None, width=0.004,
            headwidth=3.5, headlength=4, alpha=0.9, zorder=2)

        # Круги Эйнштейна
        t = np.linspace(0, 2 * np.pi, 200)
        self.ax.plot(x1 + self.te1 * np.cos(t), self.te1 * np.sin(t),
                     "-", color="cyan", linewidth=2.5, alpha=0.9, zorder=10)
        self.ax.plot(x2 + self.te2 * np.cos(t), self.te2 * np.sin(t),
                     "-", color="cyan", linewidth=2.5, alpha=0.9, zorder=10)

    def show(self):
        plt.show()


explorer = DualLensGradientExplorer()
explorer.show()
