import math
import unittest

from chebyshev_polynomial import chebyshev_t, chebyshev_u


def _approx(a, b, tol=1e-12):
    return abs(a - b) <= tol * (1.0 + abs(a) + abs(b))


class TestChebyshevT(unittest.TestCase):
    def test_t0_is_one(self):
        self.assertEqual(chebyshev_t(0, 0.7), 1.0)

    def test_t1_is_x(self):
        self.assertTrue(_approx(chebyshev_t(1, 0.3), 0.3))

    def test_t2(self):
        # T_2(x) = 2x^2 - 1
        self.assertTrue(_approx(chebyshev_t(2, 0.5), 2 * 0.25 - 1))

    def test_t3(self):
        # T_3(x) = 4x^3 - 3x
        x = 0.6
        self.assertTrue(_approx(chebyshev_t(3, x), 4 * x ** 3 - 3 * x))

    def test_t_at_one(self):
        self.assertTrue(_approx(chebyshev_t(10, 1.0), 1.0))

    def test_t_at_minus_one(self):
        # T_n(-1) = (-1)^n
        self.assertTrue(_approx(chebyshev_t(7, -1.0), -1.0))
        self.assertTrue(_approx(chebyshev_t(8, -1.0), 1.0))

    def test_t_at_zero(self):
        # T_n(0) = 0 for odd n, (-1)^{n/2} for even n
        self.assertTrue(_approx(chebyshev_t(5, 0.0), 0.0))
        self.assertTrue(_approx(chebyshev_t(6, 0.0), -1.0))

    def test_t_matches_cosine_form(self):
        # T_n(x) = cos(n arccos x) for x in [-1, 1]
        x = 0.83
        for n in range(0, 15):
            with self.subTest(n=n):
                expected = math.cos(n * math.acos(x))
                self.assertTrue(_approx(chebyshev_t(n, x), expected, tol=1e-10))

    def test_t_outside_unit_interval(self):
        # T_n(x) = cosh(n arccosh x) for x > 1
        x = 1.5
        for n in range(0, 12):
            with self.subTest(n=n):
                expected = math.cosh(n * math.acosh(x))
                self.assertTrue(_approx(chebyshev_t(n, x), expected, tol=1e-10))

    def test_t_accepts_int_x(self):
        self.assertTrue(_approx(chebyshev_t(4, 1), 1.0))

    def test_t_negative_n_raises(self):
        with self.assertRaises(ValueError):
            chebyshev_t(-1, 0.5)

    def test_t_non_int_n_raises(self):
        with self.assertRaises(TypeError):
            chebyshev_t(2.0, 0.5)

    def test_t_bool_n_raises(self):
        with self.assertRaises(TypeError):
            chebyshev_t(True, 0.5)


class TestChebyshevU(unittest.TestCase):
    def test_u0_is_one(self):
        self.assertEqual(chebyshev_u(0, 0.7), 1.0)

    def test_u1_is_2x(self):
        self.assertTrue(_approx(chebyshev_u(1, 0.3), 0.6))

    def test_u2(self):
        # U_2(x) = 4x^2 - 1
        self.assertTrue(_approx(chebyshev_u(2, 0.5), 4 * 0.25 - 1))

    def test_u3(self):
        # U_3(x) = 8x^3 - 4x
        x = 0.6
        self.assertTrue(_approx(chebyshev_u(3, x), 8 * x ** 3 - 4 * x))

    def test_u_at_one(self):
        # U_n(1) = n + 1
        self.assertTrue(_approx(chebyshev_u(10, 1.0), 11.0))

    def test_u_at_minus_one(self):
        # U_n(-1) = (n+1) * (-1)^n
        self.assertTrue(_approx(chebyshev_u(7, -1.0), 8.0 * (-1) ** 7))
        self.assertTrue(_approx(chebyshev_u(8, -1.0), 9.0 * (-1) ** 8))

    def test_u_at_zero(self):
        # U_n(0) = 0 for odd n, (-1)^{n/2} for even n
        self.assertTrue(_approx(chebyshev_u(5, 0.0), 0.0))
        self.assertTrue(_approx(chebyshev_u(6, 0.0), -1.0))

    def test_u_matches_sine_form(self):
        # U_n(x) = sin((n+1) arccos x) / sin(arccos x) for x in (-1, 1)
        x = 0.83
        theta = math.acos(x)
        for n in range(0, 15):
            with self.subTest(n=n):
                expected = math.sin((n + 1) * theta) / math.sin(theta)
                self.assertTrue(_approx(chebyshev_u(n, x), expected, tol=1e-10))

    def test_u_outside_unit_interval(self):
        # U_n(x) = sinh((n+1) arccosh x) / sinh(arccosh x) for x > 1
        x = 1.5
        a = math.acosh(x)
        for n in range(0, 12):
            with self.subTest(n=n):
                expected = math.sinh((n + 1) * a) / math.sinh(a)
                self.assertTrue(_approx(chebyshev_u(n, x), expected, tol=1e-10))

    def test_u_negative_n_raises(self):
        with self.assertRaises(ValueError):
            chebyshev_u(-1, 0.5)

    def test_u_non_int_n_raises(self):
        with self.assertRaises(TypeError):
            chebyshev_u(2.0, 0.5)


class TestRelationTU(unittest.TestCase):
    def test_derivative_relation(self):
        # d/dx T_n(x) = n U_{n-1}(x)
        x = 0.64
        h = 1e-6
        for n in range(1, 12):
            with self.subTest(n=n):
                deriv = (chebyshev_t(n, x + h) - chebyshev_t(n, x - h)) / (2 * h)
                self.assertTrue(_approx(deriv, n * chebyshev_u(n - 1, x), tol=1e-6))


if __name__ == "__main__":
    unittest.main()
