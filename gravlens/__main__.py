"""
python -m gravlens         — запуск интерактивной визуализации
python -m gravlens demo    — демо: потенциалы и карты для разных моделей
"""

import sys


def run_interactive():
    from .interactive import LensingExplorer
    explorer = LensingExplorer()
    explorer.show()


def run_demo():
    import matplotlib.pyplot as plt
    from .models import PointMass, SIS, CompositeLens
    from .plotting import plot_potential, plot_magnification_map, plot_critical_and_caustic
    from .cosmology import Cosmology

    # 1. Точечная линза
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lens = PointMass()
    plot_potential(lens, ax=axes[0], colorbar=False)
    axes[0].set_title("Потенциал: точечная масса")

    plot_magnification_map(lens, ax=axes[1], colorbar=False)
    axes[1].set_title("Усиление: точечная масса")

    # 2. Двойная точечная линза
    double = CompositeLens(
        PointMass(x0=-0.15, y0=0),
        PointMass(x0=0.15, y0=0),
    )
    plot_potential(double, grid_range=0.7, ax=axes[2], colorbar=False)
    axes[2].set_title("Потенциал: двойная линза")

    plt.tight_layout()
    plt.savefig("gravlens_demo.png", dpi=150, bbox_inches="tight")
    print("Сохранено: gravlens_demo.png")

    # 3. Космология
    cosmo = Cosmology(H0=70, omega_m=0.3)
    theta_e = cosmo.einstein_radius_solar(1e12, z_lens=0.5, z_source=1.0)
    print(f"θ_E для 10¹² M☉ (z_l=0.5, z_s=1.0): {theta_e:.3f}\"")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        run_interactive()
