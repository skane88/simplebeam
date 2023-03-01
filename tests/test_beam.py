"""
Test for methods of the Beam class.
"""

from math import isclose, nextafter

import numpy as np

from simplebeam import fix_ended, point, simple, udl


def test_moment_curve():
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

    x, y = beam.moment_curve(user_points=user_points, fast=False)

    for e in expected_points:
        assert e in x

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x

    assert isclose(max(y), 0.125 * w * l**2)

    beam = fix_ended(length=l, loads=l1)

    x, y = beam.moment_curve(user_points=user_points, fast=False)

    for e in expected_points:
        assert e in x

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x

    assert isclose(max(y), (1 / 24) * w * l**2)
    assert isclose(min(y), -(1 / 12) * w * l**2)

    P = 1
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    x, y = beam.moment_curve(user_points=user_points, fast=False)

    assert isclose(max(y), P * l * 0.25)
    assert isclose(y[x.index(0.25 * l)], 0.5 * P * l * 0.25)
    assert isclose(y[x.index(0.75 * l)], 0.5 * P * l * 0.25)

    for e in expected_points:
        assert e in x

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x


def test_shear_curve():
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

    x, y = beam.shear_curve(user_points=user_points, fast=False)

    for e in expected_points:
        assert e in x

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x

    assert isclose(max(y), 0.5 * w * l)
    assert isclose(min(y), -0.5 * w * l)

    beam = fix_ended(length=l, loads=l1)

    x, y = beam.shear_curve(user_points=user_points, fast=False)

    for e in expected_points:
        assert e in x

    assert isclose(max(y), 0.5 * w * l)
    assert isclose(min(y), -0.5 * w * l)

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x

    P = 1
    l1 = point(magnitude=P, position=l / 2)

    beam = simple(length=l, loads=l1)

    x, y = beam.shear_curve(user_points=user_points, fast=False)

    assert isclose(max(y), 0.5 * P)
    assert isclose(min(y), -0.5 * P)
    assert isclose(y[x.index(nextafter(0, l))], 0.5 * P)
    assert isclose(y[x.index(nextafter(l, 0))], -0.5 * P)
    assert isclose(y[x.index(0.25 * l)], 0.5 * P)
    assert isclose(y[x.index(0.75 * l)], -0.5 * P)

    for e in expected_points:
        assert e in x

    assert nextafter(0, l) in x
    assert nextafter(l, 0) in x


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
