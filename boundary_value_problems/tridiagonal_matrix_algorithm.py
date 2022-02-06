import numpy as np


def solve_tridiagonal_matrix_system(
    bottom_diagonal: np.ndarray,
    main_diagonal: np.ndarray,
    top_diagonal: np.ndarray,
    scalars: np.ndarray
) -> np.ndarray:
    a, b, c, d = map(np.copy, (
        bottom_diagonal,
        main_diagonal,
        top_diagonal,
        scalars
    ))

    dimension, *_ = b.shape

    for i in range(1, dimension):
        w = a[i - 1] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    x = np.copy(b)
    x[-1] = d[-1] / b[-1]

    for i in range(dimension - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x
