# chebyshev-polynomial

Evaluates Chebyshev polynomials of the first kind (T_n) and second kind (U_n) at a real point using the three-term recurrence.

## Usage

```python
from chebyshev_polynomial import chebyshev_t, chebyshev_u

# T_5(0.3) and U_5(0.3)
print(chebyshev_t(5, 0.3))
print(chebyshev_u(5, 0.3))
```

Both functions take a non-negative integer degree `n` and a real `x` (int or float), and return a `float`.

## Why this exists

The library provides a dependency-free way to evaluate Chebyshev polynomials for approximation and interpolation work where you already have the coefficients and just need values. The three-term recurrence is used instead of the closed-form `cos(n arccos x)` because the recurrence is numerically stable across the whole interval [-1, 1] and avoids catastrophic cancellation near the endpoints where `arccos` is flat.

The trade-off: the recurrence is O(n) per evaluation. If you need to evaluate at many points for the same n, or need the full set T_0..T_n at one point, a vectorised or memoised approach would be faster. This library does not do that; it evaluates one (n, x) pair at a time.

## Edge cases

- `n` must be a non-negative `int`. Booleans are rejected even though `isinstance(True, int)` is true in Python, because accepting `True` as degree 1 is a common source of silent bugs. `float` degrees like `3.0` are also rejected.
- For `|x| > 1` the polynomials grow exponentially with n. The recurrence still computes a value, but floating-point error will dominate for large n in that region. This is inherent to the problem, not a bug in the recurrence.
- Results are always `float`, including `T_0` which returns `1.0` rather than `1`.
