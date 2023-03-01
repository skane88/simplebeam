"""
Test methods in loads.py
"""

from math import isclose

from simplebeam import cantilever, simple, triangular, udl


def test_udl():
    """
    Test the UDL load helper method
    """

    length = 4
    w = 4

    load = udl(magnitude=4)

    beam = simple(length=length, loads=load)

    peak_moment = 0.125 * w * length**2

    assert isclose(beam.moment_at_point(position=length / 2), peak_moment)
    assert isclose(beam._load_at_point(position=length / 4), w)  # pylint: disable=W0212


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
