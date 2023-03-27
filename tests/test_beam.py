"""
Test for methods of the Beam class.
"""

from math import isclose, nextafter

import numpy as np
import pytest

from simplebeam import fix_ended, point, simple, udl


def check_expected_points(expected_points, actual_points, length):
    """
    Helper method to check that all expected points are found in the results.

    :param expected_points: the expected points to find.
    :param actual_points: the actual points to check.
    :param length: the length of the beam.
    """

    for e in expected_points:
        assert e in actual_points

    # Check that there is a point at the nearest possible point to 0, and to l.
    assert nextafter(0, length) in actual_points
    assert nextafter(length, 0) in actual_points


@pytest.mark.parametrize("fast", [(True), (False)])
def test_moment_curve(fast):
    """
    Test the moment curve method.

    Note that because the underlying method that approximates the curve includes a
    random choice of points, we will use the user_points function,
    along with maximum and minimum checks to determine if the method is working well.
    """

    l = 4
    w = 4

    l1 = udl(magnitude=w)

    beam = simple(length=l, loads=l1)

    user_points = [l / 8, l / 3, l / 2]
    expected_points = set(
        user_points + [nextafter(0, l), nextafter(l, 0)] + list(np.linspace(0, l, 101))
    )

    x, y = beam.moment_curve(user_points=user_points, fast=fast)

    check_expected_points(expected_points=expected_points, actual_points=x, length=l)

    assert isclose(max(y), 0.125 * w * l**2)

    beam = fix_ended(length=l, loads=l1)

    x, y = beam.moment_curve(user_points=user_points, fast=fast)

    check_expected_points(expected_points=expected_points, actual_points=x, length=l)

    assert isclose(max(y), (1 / 24) * w * l**2)
    assert isclose(min(y), -(1 / 12) * w * l**2)

    P = 1
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    x, y = beam.moment_curve(user_points=user_points, fast=fast)

    assert isclose(max(y), P * l * 0.25)
    assert isclose(y[x.index(0.25 * l)], 0.5 * P * l * 0.25)
    assert isclose(y[x.index(0.75 * l)], 0.5 * P * l * 0.25)

    check_expected_points(expected_points=expected_points, actual_points=x, length=l)


@pytest.mark.parametrize("fast", [(True), (False)])
def test_shear_curve(fast):
    """
    Test the shear curve method.

    Note that because the underlying method that approximates the curve includes a
    random choice of points, we will use the user_points function,
    along with maximum and minimum checks to determine if the method is working well.
    """

    l = 4
    w = 4

    l1 = udl(magnitude=w)

    beam = simple(length=l, loads=l1)

    user_points = [l / 8, l / 3, l / 2]
    expected_points = set(
        user_points + [nextafter(0, l), nextafter(l, 0)] + list(np.linspace(0, l, 101))
    )

    x, y = beam.shear_curve(user_points=user_points, fast=fast)

    check_expected_points(expected_points, x, l)

    assert isclose(max(y), 0.5 * w * l)
    assert isclose(min(y), -0.5 * w * l)

    beam = fix_ended(length=l, loads=l1)

    x, y = beam.shear_curve(user_points=user_points, fast=fast)

    check_expected_points(expected_points=expected_points, actual_points=x, length=l)

    assert isclose(max(y), 0.5 * w * l)
    assert isclose(min(y), -0.5 * w * l)

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x

    P = 1
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    x, y = beam.shear_curve(user_points=user_points, fast=fast)

    assert isclose(max(y), 0.5 * P)
    assert isclose(min(y), -0.5 * P)
    assert isclose(y[x.index(nextafter(0, l))], 0.5 * P)
    assert isclose(y[x.index(nextafter(l, 0))], -0.5 * P)
    assert isclose(y[x.index(0.25 * l)], 0.5 * P)
    assert isclose(y[x.index(0.75 * l)], -0.5 * P)

    check_expected_points(expected_points, x, l)


def test_shear_at_point():
    """
    Test the shear at point method.
    """

    l = 4
    P = 1
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    assert isclose(0.5, beam.shear_at_point(0))
    assert isclose(-0.5, beam.shear_at_point(l))


def test_min_result():
    """
    Test the maximum result calculations
    """

    l = 5
    P = -5000
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    E = beam.elastic_modulus
    I = beam.second_moment

    assert isclose(P / 2, beam.min_shear())
    assert isclose(P * l / 4, beam.min_moment())
    assert isclose(-0.00039062499999999997, beam.min_slope())
    assert isclose(-0.0006510416666666666, beam.min_deflection())


def test_max_result():
    """
    Test the maximum result calculations.
    """

    l = 5
    P = -5000
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    assert isclose(-P / 2, beam.max_shear())
    assert isclose(0, beam.max_moment(), abs_tol=1e-15)
    assert isclose(0.00039062499999999997, beam.max_slope())
    assert isclose(0, beam.max_deflection(), abs_tol=1e-15)
