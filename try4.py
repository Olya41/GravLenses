"""
Интерактивные эквипотенциали двух линз.

Переключение: точечные массы / SIS.
Слайдеры: theta_E1, theta_E2, расстояние.
Круги Эйнштейна + эквипотенциальные линии.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


class DualLensContourExplorer:

    def __init__(self):
        self.te1 = 1.0
        self.te2 = 1.0
        self.sep = 0.3
        self.model = "Point Mass"

        self.grid_range = 2.5
        self.N = 500

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

    def _potential_point_mass(self, r, te):
        with np.errstate(divide="ignore", invalid="ignore"):
            return te**2 * np.log(r)

    def _potential_sis(self, r, te):
        return te * r

    def _draw(self):
        d = self.sep / 2
        x1, x2 = -d, d

        x = np.linspace(-self.grid_range, self.grid_range, self.N)
        y = np.linspace(-self.grid_range, self.grid_range, self.N)
        X, Y = np.meshgrid(x, y)

        r1 = np.sqrt((X - x1)**2 + Y**2)
        r2 = np.sqrt((X - x2)**2 + Y**2)

        mask_radius = 0.015
        mask = (r1 < mask_radius) | (r2 < mask_radius)

        if self.model == "Point Mass":
            psi = self._potential_point_mass(r1, self.te1) + self._potential_point_mass(r2, self.te2)
            title = (r"Point Mass:  $\psi = \theta_{E1}^2 \ln r_1 + \theta_{E2}^2 \ln r_2$"
                     r"$\qquad (\theta_E^2 \propto M)$")
        else:
            psi = self._potential_sis(r1, self.te1) + self._potential_sis(r2, self.te2)
            title = (r"SIS:  $\psi = \theta_{E1}\, r_1 + \theta_{E2}\, r_2$"
                     r"$\qquad (\theta_E \propto \sigma_v^2)$")

        psi_masked = np.ma.array(psi, mask=mask)

        # --- Перерисовка ---
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel(r"$\theta_x$", fontsize=13)
        self.ax.set_ylabel(r"$\theta_y$", fontsize=13)
        self.ax.set_title(title, fontsize=13)
        self.ax.set_xlim(-self.grid_range, self.grid_range)
        self.ax.set_ylim(-self.grid_range, self.grid_range)
        self.ax.set_facecolor("#0e0e1a")

        # Эквипотенциали
        vmin, vmax = float(psi_masked.min()), float(psi_masked.max())
        if vmax - vmin > 1e-10:
            levels = np.linspace(vmin, vmax, 40)
            cs = self.ax.contour(
                X, Y, psi_masked, levels=levels,
                cmap="coolwarm", linewidths=1.2, zorder=2)
            self.ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f",
                           colors="white")

        # Круги Эйнштейна
        t = np.linspace(0, 2 * np.pi, 200)
        self.ax.plot(x1 + self.te1 * np.cos(t), self.te1 * np.sin(t),
                     "-", color="cyan", linewidth=2.5, alpha=0.9, zorder=10)
        self.ax.plot(x2 + self.te2 * np.cos(t), self.te2 * np.sin(t),
                     "-", color="cyan", linewidth=2.5, alpha=0.9, zorder=10)

    def show(self):
        plt.show()


explorer = DualLensContourExplorer()
explorer.show()
