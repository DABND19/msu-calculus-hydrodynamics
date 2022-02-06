from cmath import inf
from typing import Tuple

import numpy as np

from boundary_value_problems.tridiagonal_matrix_algorithm import solve_tridiagonal_matrix_system


def solve_second_order_lde(
    A: float, 
    B: float, 
    C: float, 
    *, 
    x: np.ndarray = np.linspace(0, 1, 1000), 
    Y_left: float, 
    Y_right: float, 
    precision: float = 1e-3, 
    max_iterations_count: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y'' + A * Y' + B * Y + C = 0
    """
    h = (x[-1] - x[0]) / len(x)

    Y = np.array([Y_left + (Y_right - Y_left) * x])

    norm = +inf

    while norm > precision and len(Y) <= max_iterations_count:
        Y_prev = Y[-1]

        a = (1 - 0.5 * h * A) * np.ones((len(x) - 1, ))
        b = (B * h ** 2 - 2) * np.ones((len(x), ))
        c = (1 + 0.5 * h * A) * np.ones((len(x) - 1, ))

        d = -C * h ** 2 * np.ones((len(x), ))
        d[0] = d[0] - (1 - 0.5 * h * A) * Y_left
        d[-1] = d[-1] - (1 + 0.5 * h * A) * Y_right

        Y_current = solve_tridiagonal_matrix_system(a, b, c, d)

        norm = np.max(np.abs(Y_current - Y_prev))
        Y = np.append(Y, [Y_current], axis=0)

    return x, Y
