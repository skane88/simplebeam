"""
Test methods in loads.py
"""

from math import isclose

from simplebeam import cantilever, triangular


def test_triangular():
    """
    Test the triangular load helper method
    """

    length = 4.0
    magnitude = 10.0

    expected_moment = -20 * (4 / 3)

    load = triangular(magnitude=magnitude, start=0, load_length=length)

    beam = cantilever(length=length, loads=load, fixed_left=False)

    assert isclose(expected_moment, beam.moment_at_point(length))
