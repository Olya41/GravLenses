"""
Интерактивная визуализация гравитационного линзирования.

Режим «Эллипс»: перетаскивание двигает источник.
Режим «Небо»:  перетаскивание двигает Небо.png относительно сетки;
               масштаб неба — слайдер «Масштаб неба».
Кнопка записи → .mp4 в папку animations/.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

from .solvers import point_mass_images
from .magnification import (
    shoelace_area,
    pogson_delta_m,
    generate_ellipse_boundary,
    is_inside_ellipse,
)

_STYLE = {
    # Оба изображения точечной линзы — зелёные; оба SIS — оранжевые
    "PointMass": {"img1": "forestgreen", "img2": "forestgreen", "marker": "o"},
    "SIS":       {"img1": "darkorange", "img2": "darkorange", "marker": "s"},
}

_SKY_FILE = "Небо.png"

# Правая колонка: правый край ≈ 0.99; слайдеры чуть ниже/уже — дальше от графика
_RX, _RW = 0.622, 0.368
_SH = 0.015
_SLIDER_PAD = 0.032  # отступ ползунка от левого края колонки


def _sis_images(
    beta_x: float, beta_y: float, theta_e: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.sqrt(beta_x**2 + beta_y**2)
    if beta < 1e-15:
        return np.array([theta_e, -theta_e]), np.array([0.0, 0.0])
    cos_phi = beta_x / beta
    sin_phi = beta_y / beta
    imgs_x = np.array([(beta + theta_e) * cos_phi, (beta - theta_e) * cos_phi])
    imgs_y = np.array([(beta + theta_e) * sin_phi, (beta - theta_e) * sin_phi])
    return imgs_x, imgs_y


class LensingExplorer:
    """
    Интерактивная визуализация гравитационной линзы.

        explorer = LensingExplorer()
        explorer.show()
    """

    _MODELS = ["PointMass", "SIS"]

    def __init__(self, theta_e: float = 1.0, grid_lim: float = 3.0):
        self.theta_e = theta_e
        self.grid_lim = grid_lim
        self.active_models: set[str] = {"PointMass"}

        self.semi_a = 0.25
        self.semi_b = 0.20
        self.n_boundary = 500
        self.angle_deg = 0.0

        self.ring_visible = True
        self.boundary_visible = True
        self.images_visible = True

        self.mode: str = "ellipse"  # "ellipse" | "sky"

        self.sky_x: float = 0.0
        self.sky_y: float = 0.0
        self.sky_scale: float = grid_lim * 2.0
        self.sky_alpha: float = 0.5

        self._drag_start: tuple[float, float] | None = None
        self._drag_sky0: tuple[float, float] = (0.0, 0.0)

        self.recording = False
        self._frames: list[np.ndarray] = []

        self._build_figure()
        self._build_mode_radio()
        self._build_color_legend()
        self._build_ellipse_sliders()
        self._build_sky_sliders()
        self._build_visibility_checkboxes()
        self._build_model_checkboxes()
        self._build_record_button()
        self._connect_events()

        self.dragging = False
        self.update_all(1.0, 0.0)

    # ------------------------------------------------------------------ layout

    def _build_figure(self):
        self.fig = plt.figure(figsize=(20, 13), dpi=100)

        # График слева — чуть уже, чтобы ползунки визуально не лезли на картинку
        self.ax = self.fig.add_axes([0.045, 0.07, 0.512, 0.88])
        self.ax.set_xlim(-self.grid_lim, self.grid_lim)
        self.ax.set_ylim(-self.grid_lim, self.grid_lim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3, zorder=2)
        self.ax.set_title("Гравитационное линзирование", fontsize=16, pad=10)

        self._sky_im = None
        self._sky_aspect: float = 1.0
        if os.path.exists(_SKY_FILE):
            raw = plt.imread(_SKY_FILE)
            sky_rgb = raw[..., :3] if raw.ndim == 3 and raw.shape[2] >= 3 else raw
            self._sky_img = sky_rgb
            h, w = sky_rgb.shape[:2]
            self._sky_aspect = w / h
            self._sky_im = self.ax.imshow(
                sky_rgb,
                extent=self._sky_extent(),
                origin="upper",
                alpha=self.sky_alpha,
                zorder=1,
            )
            self.ax.set_aspect("equal")
        else:
            self._sky_img = None

        # Под слайдерами неба, не пересекается с ними и с галочками
        self.info_ax = self.fig.add_axes([_RX, 0.208, _RW, 0.10])
        self.info_ax.set_axis_off()
        self.info_text = self.info_ax.text(
            0.05, 0.98, "", transform=self.info_ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
            visible=False,
        )

        self.einstein_circle = Circle(
            (0, 0), self.theta_e, fill=False,
            color="blue", linewidth=1.5, linestyle="--", alpha=0.8, zorder=5,
        )
        self.ax.add_patch(self.einstein_circle)

        self.source_dot = Circle((1, 0), 0.025, color="red", alpha=0.9, zorder=10)
        self.ax.add_patch(self.source_dot)
        self.source_ellipse = Ellipse(
            (1, 0), width=2 * self.semi_a, height=2 * self.semi_b,
            fill=False, color="red", linewidth=1.0, alpha=0.5, zorder=10,
        )
        self.ax.add_patch(self.source_ellipse)

        self._img_dots: dict[str, list[Circle]] = {}
        self._lines: dict[str, list] = {}
        for model in self._MODELS:
            st = _STYLE[model]
            d1 = Circle((0, 0), 0.025, color=st["img1"], alpha=0.9, zorder=10)
            d2 = Circle((0, 0), 0.025, color=st["img2"], alpha=0.9, zorder=10)
            self.ax.add_patch(d1)
            self.ax.add_patch(d2)
            d1.set_visible(False)
            d2.set_visible(False)
            self._img_dots[model] = [d1, d2]
            l1, = self.ax.plot([], [], "--", color="gray", lw=0.5, alpha=0.2, zorder=3)
            l2, = self.ax.plot([], [], "--", color="gray", lw=0.5, alpha=0.2, zorder=3)
            self._lines[model] = [l1, l2]

        self._scatters: dict[str, list] = {m: [None, None] for m in self._MODELS}
        self.scatter_src = None

        self._src_bx = self._src_by = np.empty(0)
        self._f1x = self._f1y = self._f2x = self._f2y = np.empty(0)
        self._info_model: str = "PointMass"

    def _build_mode_radio(self):
        ax_r = self.fig.add_axes([_RX, 0.912, _RW, 0.054])
        ax_r.set_facecolor("#f0f4f8")
        ax_r.set_title("Режим мыши", fontsize=8.5, pad=1)
        self.radio_mode = RadioButtons(
            ax_r,
            ["  Двигать эллипс", "  Двигать небо"],
            active=0, activecolor="steelblue",
        )
        for label in self.radio_mode.labels:
            label.set_fontsize(9.5)
        self.radio_mode.on_clicked(self._on_mode_change)

    def _build_color_legend(self):
        ax = self.fig.add_axes([_RX, 0.688, _RW, 0.198])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.0, 0.98, "Обозначения", fontsize=11, fontweight="bold", va="top")
        items = [
            ("red",                        "o",  "Источник"),
            (_STYLE["PointMass"]["img1"],   "o",  "PM: 1"),
            (_STYLE["PointMass"]["img2"],   "o",  "PM: 2"),
            (_STYLE["SIS"]["img1"],         "s",  "SIS: 1"),
            (_STYLE["SIS"]["img2"],         "s",  "SIS: 2"),
            ("blue",                        "--", "Кольцо Эйнштейна"),
        ]
        for i, (color, marker, label) in enumerate(items):
            y = 0.82 - i * 0.128
            if marker == "--":
                ax.plot([0.02, 0.12], [y, y], "--",
                        color=color, lw=1.8, alpha=0.8, solid_capstyle="round")
            else:
                ax.plot(0.06, y, marker=marker, color=color, ms=8, linestyle="none")
            ax.text(0.16, y, label, va="center", fontsize=10)

    def _build_ellipse_sliders(self):
        """Источник — правая колонка, под легендой."""
        sl = _RX + _SLIDER_PAD * 0.35
        sw, sh = _RW - _SLIDER_PAD, _SH
        y0 = 0.572
        step = 0.022
        self.fig.text(sl + sw / 2, y0 + 0.037, "Источник",
                      ha="center", fontsize=8.5, color="#333", fontweight="bold")

        ys = [y0, y0 - step, y0 - 2 * step, y0 - 3 * step]
        self.slider_angle = Slider(
            self.fig.add_axes([sl, ys[0], sw, sh]),
            "Угол °", -90, 90, valinit=0)
        self.slider_a = Slider(
            self.fig.add_axes([sl, ys[1], sw, sh]),
            "Полуось A", 0.05, 0.5, valinit=self.semi_a)
        self.slider_b = Slider(
            self.fig.add_axes([sl, ys[2], sw, sh]),
            "Полуось B", 0.05, 0.5, valinit=self.semi_b)
        self.slider_n = Slider(
            self.fig.add_axes([sl, ys[3], sw, sh]),
            "Точки", 100, 5000, valinit=self.n_boundary, valstep=50)

        for s in (self.slider_angle, self.slider_a, self.slider_b, self.slider_n):
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(8)

        self.slider_angle.on_changed(lambda v: self._on_param_change(angle=v))
        self.slider_a.on_changed(lambda v: self._on_param_change(a=v))
        self.slider_b.on_changed(lambda v: self._on_param_change(b=v))
        self.slider_n.on_changed(lambda v: self._on_param_change(n=int(v)))

    def _build_sky_sliders(self):
        """Небо — правая колонка, ниже источника."""
        sl = _RX + _SLIDER_PAD * 0.35
        sw, sh = _RW - _SLIDER_PAD, _SH
        lim = self.grid_lim
        y0 = 0.422
        step = 0.022
        self.fig.text(sl + sw / 2, y0 + 0.037, "Небо (относительно сетки)",
                      ha="center", fontsize=8.5, color="#333", fontweight="bold")

        ys = [y0, y0 - step, y0 - 2 * step, y0 - 3 * step]
        self.slider_sky_alpha = Slider(
            self.fig.add_axes([sl, ys[0], sw, sh]),
            "Прозрачность", 0.0, 1.0, valinit=self.sky_alpha)
        # Масштаб: высота картинки в единицах сетки (меньше — сильнее «зум» неба)
        self.slider_sky_scale = Slider(
            self.fig.add_axes([sl, ys[1], sw, sh]),
            "Масштаб неба", 0.15, lim * 8, valinit=self.sky_scale)
        self.slider_sky_y = Slider(
            self.fig.add_axes([sl, ys[2], sw, sh]),
            "y неба", -lim * 3, lim * 3, valinit=self.sky_y)
        self.slider_sky_x = Slider(
            self.fig.add_axes([sl, ys[3], sw, sh]),
            "x неба", -lim * 3, lim * 3, valinit=self.sky_x)

        for s in (self.slider_sky_x, self.slider_sky_y,
                  self.slider_sky_scale, self.slider_sky_alpha):
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(8)

        self.slider_sky_x.on_changed(self._on_sky_slider)
        self.slider_sky_y.on_changed(self._on_sky_slider)
        self.slider_sky_scale.on_changed(self._on_sky_slider)
        self.slider_sky_alpha.on_changed(self._on_sky_slider)

    def _build_visibility_checkboxes(self):
        ax_cb = self.fig.add_axes([_RX, 0.095, 0.17, 0.095])
        ax_cb.set_title("Показать", fontsize=9, pad=1)
        self.cb_vis = CheckButtons(
            ax_cb, ["Кольцо", "Источник", "Изобр."], [True, True, True])
        for label in self.cb_vis.labels:
            label.set_fontsize(9)
        self.cb_vis.on_clicked(self._on_visibility_clicked)

    def _build_model_checkboxes(self):
        ax_cb = self.fig.add_axes([_RX + 0.195, 0.095, 0.175, 0.095])
        ax_cb.set_title("Модель", fontsize=9, pad=1)
        self.cb_model = CheckButtons(ax_cb, self._MODELS, [True, False])
        for label in self.cb_model.labels:
            label.set_fontsize(9)
        self.cb_model.on_clicked(self._on_model_toggle)

    def _build_record_button(self):
        ax_btn = self.fig.add_axes([_RX + 0.012, 0.028, _RW - 0.024, 0.047])
        self.btn_rec = Button(ax_btn, "●  Начать запись",
                              color="mistyrose", hovercolor="lightcoral")
        self.btn_rec.label.set_fontsize(9.5)
        self.btn_rec.on_clicked(self._on_record_toggle)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # --------------------------------------------------------------- callbacks

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.mode == "ellipse":
            if self.source_dot.contains(event)[0]:
                self.dragging = True
        else:
            self.dragging = True
            self._drag_start = (event.xdata, event.ydata)
            self._drag_sky0 = (self.sky_x, self.sky_y)

    def _on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        if self.mode == "ellipse":
            self.update_all(event.xdata, event.ydata)
        else:
            if self._drag_start is None:
                return
            dx = event.xdata - self._drag_start[0]
            dy = event.ydata - self._drag_start[1]
            self.sky_x = self._drag_sky0[0] + dx
            self.sky_y = self._drag_sky0[1] + dy
            self.slider_sky_x.eventson = False
            self.slider_sky_y.eventson = False
            self.slider_sky_x.set_val(
                np.clip(self.sky_x, self.slider_sky_x.valmin, self.slider_sky_x.valmax))
            self.slider_sky_y.set_val(
                np.clip(self.sky_y, self.slider_sky_y.valmin, self.slider_sky_y.valmax))
            self.slider_sky_x.eventson = True
            self.slider_sky_y.eventson = True
            self._update_sky()
            self.ax.set_aspect("equal")
        self.fig.canvas.draw()
        if self.recording:
            self._capture_frame()

    def _on_release(self, event):
        self.dragging = False
        self._drag_start = None

    def _on_mode_change(self, label):
        self.mode = "sky" if "небо" in label.lower() else "ellipse"

    def _on_sky_slider(self, _val=None):
        self.sky_x = self.slider_sky_x.val
        self.sky_y = self.slider_sky_y.val
        self.sky_scale = self.slider_sky_scale.val
        self.sky_alpha = self.slider_sky_alpha.val
        self._update_sky()
        self.ax.set_aspect("equal")
        self.fig.canvas.draw_idle()

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
        if self.recording:
            self._capture_frame()

    def _on_visibility_clicked(self, label):
        if label == "Кольцо":
            self.ring_visible = not self.ring_visible
            self.einstein_circle.set_visible(self.ring_visible)
        elif label == "Источник":
            self.boundary_visible = not self.boundary_visible
            if self.scatter_src is not None:
                self.scatter_src.set_visible(self.boundary_visible)
        elif label == "Изобр.":
            self.images_visible = not self.images_visible
            self._apply_model_visibility()
        self.fig.canvas.draw_idle()

    def _on_model_toggle(self, label):
        if label in self.active_models:
            self.active_models.discard(label)
        else:
            self.active_models.add(label)
        self._apply_model_visibility()
        x, y = self.source_dot.center
        beta = np.sqrt(x**2 + y**2)
        self._update_info(x, y, beta)
        self.fig.canvas.draw_idle()

    def _apply_model_visibility(self):
        for model in self._MODELS:
            active = model in self.active_models
            for dot in self._img_dots[model]:
                if not active:
                    dot.set_visible(False)
            for sc in self._scatters[model]:
                if sc is not None:
                    sc.set_visible(active and self.images_visible)

    def _on_record_toggle(self, event):
        if not self.recording:
            self.recording = True
            self._frames = []
            self.btn_rec.label.set_text("■  Остановить запись")
            self.btn_rec.ax.set_facecolor("tomato")
            self.ax.set_title("Гравитационное линзирование  ● REC",
                               fontsize=16, pad=10, color="crimson")
        else:
            self.recording = False
            self.btn_rec.label.set_text("●  Начать запись")
            self.btn_rec.ax.set_facecolor("mistyrose")
            self.ax.set_title("Гравитационное линзирование",
                               fontsize=16, pad=10, color="black")
            self._save_recording()
        self.fig.canvas.draw_idle()

    def _capture_frame(self):
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        self._frames.append(np.asarray(buf)[..., :3].copy())

    def _save_recording(self):
        if not self._frames:
            print("Запись пустая — ничего не сохранено.")
            return
        os.makedirs("animations", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("animations", f"lensing_{ts}.mp4")
        try:
            import imageio
            with imageio.get_writer(path, fps=15, codec="libx264",
                                    quality=8, macro_block_size=1) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
            n = len(self._frames)
            print(f"Сохранено: {path}  ({n} кадров, ~{n/15:.1f} с)")
        except Exception as exc:
            print(f"Ошибка при сохранении: {exc}")
            self._save_frames_fallback(path)

    def _save_frames_fallback(self, mp4_path: str):
        base = mp4_path.replace(".mp4", "")
        os.makedirs(base, exist_ok=True)
        try:
            from PIL import Image
            for i, frame in enumerate(self._frames):
                Image.fromarray(frame).save(os.path.join(base, f"{i:04d}.png"))
            print(f"Кадры сохранены в папку: {base}/")
        except Exception as exc2:
            print(f"Fallback тоже не удался: {exc2}")

    def _sky_extent(self) -> list[float]:
        half_h = self.sky_scale / 2
        half_w = half_h * self._sky_aspect
        return [
            self.sky_x - half_w,
            self.sky_x + half_w,
            self.sky_y - half_h,
            self.sky_y + half_h,
        ]

    def _update_sky(self):
        if self._sky_im is not None:
            self._sky_im.set_extent(self._sky_extent())
            self._sky_im.set_alpha(self.sky_alpha)

    def _center_images(self, model: str, x: float, y: float):
        if model == "PointMass":
            return point_mass_images(x, y, self.theta_e)
        return _sis_images(x, y, self.theta_e)

    def _img2_exists(self, model: str, beta: float) -> bool:
        return model == "PointMass" or beta < self.theta_e

    def _boundary_images_for(
        self, model: str, bx: np.ndarray, by: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = np.sqrt(bx**2 + by**2)
        valid = r > 1e-10
        bx_v, by_v, r_v = bx[valid], by[valid], r[valid]

        if model == "PointMass":
            theta_p = r_v / 2 + np.sqrt((r_v / 2)**2 + self.theta_e**2)
            theta_m = r_v / 2 - np.sqrt((r_v / 2)**2 + self.theta_e**2)
            f1x, f1y = theta_p * bx_v / r_v, theta_p * by_v / r_v
            f2x, f2y = theta_m * bx_v / r_v, theta_m * by_v / r_v
        else:
            f1x = (r_v + self.theta_e) / r_v * bx_v
            f1y = (r_v + self.theta_e) / r_v * by_v
            mask2 = r_v < self.theta_e
            if mask2.any():
                r2 = r_v[mask2]
                f2x = (r2 - self.theta_e) / r2 * bx_v[mask2]
                f2y = (r2 - self.theta_e) / r2 * by_v[mask2]
            else:
                f2x = np.empty(0)
                f2y = np.empty(0)

        return f1x, f1y, f2x, f2y

    def update_all(self, x: float, y: float):
        self.source_dot.center = (x, y)
        self.source_ellipse.center = (x, y)
        self.source_ellipse.angle = self.angle_deg

        beta = np.sqrt(x**2 + y**2)

        for model in self._MODELS:
            active = model in self.active_models
            dots, lines = self._img_dots[model], self._lines[model]
            if active:
                imgs_x, imgs_y = self._center_images(model, x, y)
                has2 = self._img2_exists(model, beta)
                dots[0].center = (imgs_x[0], imgs_y[0])
                dots[0].set_visible(True)
                dots[1].center = (imgs_x[1], imgs_y[1])
                dots[1].set_visible(has2)
                lines[0].set_data([x, imgs_x[0]], [y, imgs_y[0]])
                lines[1].set_data(
                    [x, imgs_x[1]] if has2 else [],
                    [y, imgs_y[1]] if has2 else [],
                )
            else:
                for d in dots:
                    d.set_visible(False)
                for l in lines:
                    l.set_data([], [])

        self._update_boundary_points(x, y)
        self._update_info(x, y, beta)

    def _update_boundary_points(self, x: float, y: float):
        if self.scatter_src is not None:
            self.scatter_src.remove()
        for model in self._MODELS:
            for i, sc in enumerate(self._scatters[model]):
                if sc is not None:
                    sc.remove()
                    self._scatters[model][i] = None

        bx, by = generate_ellipse_boundary(
            x, y, self.semi_a, self.semi_b, self.n_boundary, self.angle_deg)

        self.scatter_src = self.ax.scatter(
            bx, by, c="red", s=2, alpha=0.25, zorder=4, linewidths=0)
        self.scatter_src.set_visible(self.boundary_visible)
        self._src_bx, self._src_by = bx, by

        info_set = False
        for model in self._MODELS:
            st = _STYLE[model]
            f1x, f1y, f2x, f2y = self._boundary_images_for(model, bx, by)

            sc1 = self.ax.scatter(
                f1x, f1y, c=st["img1"], s=2, alpha=0.30,
                marker=st["marker"], zorder=4, linewidths=0)
            sc1.set_visible(model in self.active_models and self.images_visible)
            self._scatters[model][0] = sc1

            if len(f2x) > 0:
                sc2 = self.ax.scatter(
                    f2x, f2y, c=st["img2"], s=2, alpha=0.30,
                    marker=st["marker"], zorder=4, linewidths=0)
                sc2.set_visible(model in self.active_models and self.images_visible)
                self._scatters[model][1] = sc2

            if not info_set:
                self._f1x, self._f1y = f1x, f1y
                self._f2x, self._f2y = f2x, f2y
                self._info_model = model
                info_set = True

    def _update_info(self, x: float, y: float, beta: float):
        model = "PointMass" if "PointMass" in self.active_models else (
            next(iter(self.active_models), self._info_model))
        self._info_model = model

        imgs_x, imgs_y = self._center_images(model, x, y)
        has2 = self._img2_exists(model, beta)

        area_src = shoelace_area(self._src_bx, self._src_by)
        einstein_area = np.pi * self.theta_e**2
        area_f1_raw = shoelace_area(self._f1x, self._f1y) if len(self._f1x) >= 3 else 0.0
        area_f2_raw = shoelace_area(self._f2x, self._f2y) if len(self._f2x) >= 3 else 0.0

        origin_inside = is_inside_ellipse(
            0, 0, x, y, self.semi_a, self.semi_b, self.angle_deg)
        if origin_inside and model == "PointMass":
            area_f1 = area_f1_raw - einstein_area
            area_f2 = einstein_area - area_f2_raw
        else:
            area_f1, area_f2 = area_f1_raw, area_f2_raw

        if area_src > 0:
            k1 = area_f1 / area_src
            k2 = (area_f2 / area_src) if has2 else 0.0
            dm1 = pogson_delta_m(k1) if k1 > 0 else 0.0
            dm2 = pogson_delta_m(k2) if (k2 > 0 and has2) else 0.0
        else:
            k1 = k2 = dm1 = dm2 = 0.0

        img2_coords = f"({imgs_x[1]:+.4f}, {imgs_y[1]:+.4f})" if has2 else "—"
        info = (
            f"Инфо: {'PM' if model == 'PointMass' else 'SIS'}\n"
            "─────────────────────────\n"
            f"  Источник  ({x:+.4f}, {y:+.4f})\n"
            f"  Изобр. 1  ({imgs_x[0]:+.4f}, {imgs_y[0]:+.4f})\n"
            f"  Изобр. 2  {img2_coords}\n\n"
            f"  S₀={area_src:.4f}  S₁={area_f1:.4f}\n"
            f"  k₁={k1:.3f}  Δm₁={dm1:+.2f}\n"
            f"  k₂={k2:.3f}  Δm₂={dm2:+.2f}\n"
        )
        self.info_text.set_text(info)

    def show(self):
        plt.show()
