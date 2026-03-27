"""
Тестирование модуля magnification.py

Проверяет:
- Вычисление площадей по формуле Гаусса (shoelace)
- Формулу Погсона для звёздных величин
- Генерацию эллипсов (граница и заполнение)
- Проверку принадлежности точки эллипсу
- Отношение потоков и полное усиление
"""

import numpy as np
import matplotlib.pyplot as plt
from gravlens import (
    shoelace_area,
    pogson_delta_m,
    flux_ratio,
    total_magnification,
    generate_ellipse_boundary,
    generate_ellipse_fill,
    is_inside_ellipse,
)


def test_shoelace_area():
    """Тест вычисления площади многоугольника."""
    print("=" * 60)
    print("ТЕСТ 1: Площадь многоугольника (формула Гаусса)")
    print("=" * 60)
    
    # Квадрат 2×2
    x_square = np.array([0, 2, 2, 0])
    y_square = np.array([0, 0, 2, 2])
    area_square = shoelace_area(x_square, y_square)
    print(f"Квадрат 2×2:")
    print(f"  Вычисленная площадь: {area_square:.6f}")
    print(f"  Ожидаемая площадь:   4.000000")
    print(f"  Ошибка: {abs(area_square - 4.0):.2e}")
    
    # Треугольник
    x_tri = np.array([0, 3, 0])
    y_tri = np.array([0, 0, 4])
    area_tri = shoelace_area(x_tri, y_tri)
    print(f"\nТреугольник (основание=3, высота=4):")
    print(f"  Вычисленная площадь: {area_tri:.6f}")
    print(f"  Ожидаемая площадь:   6.000000")
    print(f"  Ошибка: {abs(area_tri - 6.0):.2e}")
    
    # Окружность (аппроксимация многоугольником)
    n = 1000
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1.0
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    area_circle = shoelace_area(x_circle, y_circle)
    expected_circle = np.pi * r**2
    print(f"\nОкружность радиуса 1 ({n} точек):")
    print(f"  Вычисленная площадь: {area_circle:.6f}")
    print(f"  Ожидаемая площадь:   {expected_circle:.6f}")
    print(f"  Ошибка: {abs(area_circle - expected_circle):.2e}")
    print()


def test_pogson_formula():
    """Тест формулы Погсона для звёздных величин."""
    print("=" * 60)
    print("ТЕСТ 2: Формула Погсона (Δm = -2.5 lg|μ|)")
    print("=" * 60)
    
    magnifications = [1.0, 2.0, 5.0, 10.0, 100.0, 0.5, 0.1]
    
    print(f"{'μ':>8} | {'Δm':>8} | Интерпретация")
    print("-" * 60)
    for mu in magnifications:
        dm = pogson_delta_m(mu)
        if mu > 1:
            interp = f"усиление в {mu:.1f}×"
        elif mu < 1:
            interp = f"ослабление в {1/mu:.1f}×"
        else:
            interp = "без изменений"
        print(f"{mu:8.2f} | {dm:+8.3f} | {interp}")
    
    # Проверка массива
    mu_array = np.array([1.0, 2.0, 5.0, 10.0])
    dm_array = pogson_delta_m(mu_array)
    print(f"\nМассив усилений: {mu_array}")
    print(f"Массив Δm:       {dm_array}")
    print()


def test_flux_and_total():
    """Тест отношения потоков и полного усиления."""
    print("=" * 60)
    print("ТЕСТ 3: Отношение потоков и полное усиление")
    print("=" * 60)
    
    # Пример: два изображения точечной линзы
    mu1 = 5.0
    mu2 = -2.0  # отрицательное (инвертированное изображение)
    
    ratio = flux_ratio(mu1, mu2)
    total = total_magnification(np.array([mu1, mu2]))
    
    print(f"Изображение 1: μ₁ = {mu1:+.2f}")
    print(f"Изображение 2: μ₂ = {mu2:+.2f}")
    print(f"Отношение потоков: |μ₁/μ₂| = {ratio:.3f}")
    print(f"Полное усиление: Σ|μᵢ| = {total:.3f}")
    print()


def test_ellipse_generation():
    """Тест генерации эллипсов."""
    print("=" * 60)
    print("ТЕСТ 4: Генерация эллипсов")
    print("=" * 60)
    
    # Параметры эллипса
    cx, cy = 1.0, 0.5
    a, b = 0.8, 0.4
    angle_deg = 30.0
    
    # Граница
    bx, by = generate_ellipse_boundary(cx, cy, a, b, n_points=500, angle_deg=angle_deg)
    area_boundary = shoelace_area(bx, by)
    expected_area = np.pi * a * b
    
    print(f"Эллипс: центр=({cx}, {cy}), a={a}, b={b}, угол={angle_deg}°")
    print(f"Граница ({len(bx)} точек):")
    print(f"  Площадь (shoelace): {area_boundary:.6f}")
    print(f"  Ожидаемая площадь:  {expected_area:.6f}")
    print(f"  Ошибка: {abs(area_boundary - expected_area):.2e}")
    
    # Заполнение
    fx, fy = generate_ellipse_fill(cx, cy, a, b, n_points=4000, angle_deg=angle_deg)
    print(f"\nЗаполнение ({len(fx)} точек):")
    print(f"  Диапазон x: [{fx.min():.3f}, {fx.max():.3f}]")
    print(f"  Диапазон y: [{fy.min():.3f}, {fy.max():.3f}]")
    
    # Проверка принадлежности
    test_points = [
        (cx, cy, "центр"),
        (cx + 0.5*a, cy, "внутри по оси a"),
        (cx, cy + 0.5*b, "внутри по оси b"),
        (cx + 1.5*a, cy, "снаружи по оси a"),
        (cx, cy + 1.5*b, "снаружи по оси b"),
    ]
    
    print(f"\nПроверка принадлежности:")
    for px, py, desc in test_points:
        inside = is_inside_ellipse(px, py, cx, cy, a, b, angle_deg)
        status = "✓ внутри" if inside else "✗ снаружи"
        print(f"  ({px:.2f}, {py:.2f}) [{desc:20s}]: {status}")
    print()
    
    return bx, by, fx, fy, cx, cy, a, b, angle_deg


def visualize_ellipses(bx, by, fx, fy, cx, cy, a, b, angle_deg):
    """Визуализация эллипсов."""
    print("=" * 60)
    print("ТЕСТ 5: Визуализация эллипсов")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # График 1: Граница
    axes[0].plot(bx, by, 'b-', linewidth=1.5, label='Граница')
    axes[0].plot(cx, cy, 'ro', markersize=8, label='Центр')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f'Граница эллипса\n(a={a}, b={b}, угол={angle_deg}°)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    # График 2: Заполнение
    axes[1].scatter(fx, fy, c='green', s=1, alpha=0.3, label='Заполнение')
    axes[1].plot(bx, by, 'b-', linewidth=1, alpha=0.5, label='Граница')
    axes[1].plot(cx, cy, 'ro', markersize=8, label='Центр')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title(f'Заполнение эллипса\n({len(fx)} точек)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    
    # График 3: Несколько эллипсов с разными параметрами
    angles = [0, 30, 60, 90]
    colors = ['red', 'blue', 'green', 'purple']
    for angle, color in zip(angles, colors):
        ex, ey = generate_ellipse_boundary(0, 0, 1.0, 0.5, n_points=300, angle_deg=angle)
        axes[2].plot(ex, ey, color=color, linewidth=1.5, label=f'{angle}°', alpha=0.7)
    
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_title('Эллипсы с разными углами\n(a=1.0, b=0.5)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('try6_ellipses.png', dpi=150, bbox_inches='tight')
    print("Сохранено: try6_ellipses.png")
    print()


def test_magnification_scenarios():
    """Тест различных сценариев усиления."""
    print("=" * 60)
    print("ТЕСТ 6: Сценарии усиления для гравитационных линз")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Точечная линза (далёкий источник)',
            'magnifications': np.array([8.5, -1.2]),
            'description': 'Два изображения: яркое внешнее и слабое внутреннее'
        },
        {
            'name': 'Точечная линза (близкий источник)',
            'magnifications': np.array([2.1, -0.9]),
            'description': 'Источник далеко от кольца Эйнштейна'
        },
        {
            'name': 'Двойная линза',
            'magnifications': np.array([5.2, 3.8, -1.5, -0.8]),
            'description': 'Четыре изображения с разными усилениями'
        },
        {
            'name': 'Кольцо Эйнштейна',
            'magnifications': np.array([100.0, -100.0]),
            'description': 'Источник точно за линзой'
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nСценарий {i}: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"  Усиления: {scenario['magnifications']}")
        
        total = total_magnification(scenario['magnifications'])
        delta_m_values = pogson_delta_m(scenario['magnifications'])
        
        print(f"  Полное усиление: {total:.3f}")
        print(f"  Δm для каждого изображения: {delta_m_values}")
        
        # Отношения потоков между изображениями
        mags = scenario['magnifications']
        if len(mags) >= 2:
            ratio_01 = flux_ratio(mags[0], mags[1])
            print(f"  Отношение потоков (1-е/2-е): {ratio_01:.3f}")
    
    print()


def main():
    """Главная функция."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ magnification.py")
    print("=" * 60 + "\n")
    
    # Запуск тестов
    test_shoelace_area()
    test_pogson_formula()
    test_flux_and_total()
    bx, by, fx, fy, cx, cy, a, b, angle_deg = test_ellipse_generation()
    visualize_ellipses(bx, by, fx, fy, cx, cy, a, b, angle_deg)
    test_magnification_scenarios()
    
    print("=" * 60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()
