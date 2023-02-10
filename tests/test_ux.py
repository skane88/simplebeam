"""
A series of tests to test / demonstrate the UI
"""

from math import isclose

from simplebeam import Beam, fixed, guide, moment, pin, point, udl


def test_basic():
    """
    An example of defining a beam and extracting reactions etc.
    """

    l = 5.0
    E = 200e9
    I = 1.0
    P = 1.0

    # first create some restraints for the beam.
    r1 = pin(position=0)
    r2 = guide(position=l)

    # next create a load.
    load_1 = point(magnitude=P, position=l / 2)

    # now set up the beam.
    beam = Beam(
        elastic_modulus=E,
        second_moment=I,
        length=l,
        restraints=[r1, r2],
        loads=load_1,
    )

    # now the beam is set up, we need to solve it.
    beam.solve()

    # extracting reactions from the beam.
    assert isclose(beam.reactions[0]["R"], -P)
    assert beam.reactions[0]["M"] is None
    assert beam.reactions[1]["R"] is None
    assert isclose(beam.reactions[1]["M"], P * l / 2)

    # extracting shear from the beam.
    assert isclose(beam.shear_at_point(position=0.0), P)
    assert isclose(beam.shear_at_point(position=1.0), P)
    assert isclose(beam.shear_at_point(position=4.0), 0.0)
    assert isclose(beam.shear_at_point(position=l), 0.0)

    # extracting moment from the beam.
    assert isclose(beam.moment_at_point(position=0), 0, abs_tol=1e-9)
    assert isclose(beam.moment_at_point(position=l / 2), P * l / 2)
    assert isclose(beam.moment_at_point(position=l), P * l / 2)
