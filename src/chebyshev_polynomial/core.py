"""Chebyshev polynomials of the first and second kind via three-term recurrence.

The recurrence is numerically stable on [-1, 1] and avoids the explicit
trigonometric form, which loses precision near x = ±1 because it subtracts
nearly-equal angles. For |x| > 1 the recurrence still works (the polynomials are
defined for all real x), but growth is exponential and floating-point error will
dominate for large n — callers in that regime should reconsider their approach.
"""


def chebyshev_t(n, x):
    """Evaluate the Chebyshev polynomial of the first kind T_n at x.

    Uses the recurrence T_0 = 1, T_1 = x, T_{k+1} = 2 x T_k - T_{k-1}.

    Args:
        n: Non-negative integer degree.
        x: Real point at which to evaluate. May be an int or float.

    Returns:
        float value of T_n(x).

    Raises:
        TypeError: if n is not an int.
        ValueError: if n is negative.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")
    x = float(x)
    if n == 0:
        return 1.0
    if n == 1:
        return x
    t_prev, t_curr = 1.0, x
    for _ in range(2, n + 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
    return t_curr


def chebyshev_u(n, x):
    """Evaluate the Chebyshev polynomial of the second kind U_n at x.

    Uses the recurrence U_0 = 1, U_1 = 2 x, U_{k+1} = 2 x U_k - U_{k-1}.

    Args:
        n: Non-negative integer degree.
        x: Real point at which to evaluate. May be an int or float.

    Returns:
        float value of U_n(x).

    Raises:
        TypeError: if n is not an int.
        ValueError: if n is negative.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")
    x = float(x)
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    u_prev, u_curr = 1.0, 2.0 * x
    for _ in range(2, n + 1):
        u_prev, u_curr = u_curr, 2.0 * x * u_curr - u_prev
    return u_curr
