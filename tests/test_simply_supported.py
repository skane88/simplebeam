"""
Contains some tests of a simply supported beam.
"""

from math import isclose

import numpy as np
import pytest
import sympy as sym

from simplebeam import Beam, pin, point


@pytest.mark.parametrize(
    "elastic_modulus, second_moment, length, force", ((200e9, 0.1, 5.0, 5.0),)
)
def test_point_load(elastic_modulus, second_moment, length, force):
    """
    Test a simply supported beam with a point load.
    """

    r1 = pin(0)
    r2 = pin(length)

    point_load = point(magnitude=force, position=length / 2)

    test_beam = Beam(
        elastic_modulus=elastic_modulus,
        second_moment=second_moment,
        length=length,
        restraints=[r1, r2],
        loads=point_load,
    )
    test_beam.solve()

    # now define some equations for the shear, moment etc.
    l, x, P, E, I = sym.symbols("l, x, P, E, I")

    V = sym.Piecewise((P / 2, x <= l / 2), (-P / 2, x <= l))
    V = V.subs(l, length).subs(P, force)

    x_vals = list(np.linspace(0, length, 10))
    x_vals.append(length / 2 - 0.001)
    x_vals.append(length / 2 + 0.001)

    for xi in x_vals:
        assert isclose(test_beam.shear_at_point(xi), V.subs(x, xi))

    M = sym.Piecewise((P * x / 2, x <= l / 2), (P * (l - x) / 2, x <= l))
    M = M.subs(l, length).subs(P, force)

    x_vals = np.linspace(0, length, 11)

    for xi in x_vals:
        assert isclose(test_beam.moment_at_point(xi), M.subs(x, xi), abs_tol=1e-14)

    slope = {
        0: force * length**2 / (16 * elastic_modulus * second_moment),
        length / 2: 0,
        length: -force * length**2 / (16 * elastic_modulus * second_moment),
    }

    for xi, s in slope.items():
        assert isclose(test_beam.slope_at_point(xi), s, abs_tol=1e-14)

    delta = sym.Piecewise(
        (((P * x) / (48 * E * I)) * (3 * l**2 - 4 * x**2), x <= l / 2),
        (((P * (l - x)) / (48 * E * I)) * (3 * l**2 - 4 * (l - x) ** 2), x <= l),
    )
    delta = (
        delta.subs(l, length)
        .subs(P, force)
        .subs(E, elastic_modulus)
        .subs(I, second_moment)
    )

    for xi in x_vals:
        assert isclose(
            test_beam.deflection_at_point(xi), delta.subs(x, xi), abs_tol=1e-14
        )
