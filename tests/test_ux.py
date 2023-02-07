"""
A series of tests to test / demonstrate the UI
"""

from math import isclose

from simplebeam import Beam, fixed, guide, moment, pin, point, udl


def test_basic():
    """
    An example of defining a beam and extracting reactions etc.
    """

    length = 5.0

    # first create some restraints for the beam.
    r1 = pin(position=0)
    r2 = guide(position=length)

    # next create a load.
    l1 = point(magnitude=1.0, position=length / 2)

    # now set up the beam.
    beam = Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=length,
        restraints=[r1, r2],
        loads=l1,
    )

    # now the beam is set up, we need to solve it.
    beam.solve()

    # extracting reactions from the beam.
    assert isclose(beam.reactions[0]["R"], -1.0)
    assert beam.reactions[0]["M"] is None
    assert beam.reactions[1]["R"] is None
    assert isclose(beam.reactions[1]["M"], 2.5)

    # extracting shear from the beam.
    assert isclose(beam.shear_at_point(position=0.0), 1.0)
    assert isclose(beam.shear_at_point(position=1.0), 1.0)
    assert isclose(beam.shear_at_point(position=4.0), 0.0)
    assert isclose(beam.shear_at_point(position=length), 0.0)

    # extracting moment from the beam.
    assert isclose(beam.moment_at_point(position=0), 0, abs_tol=1e-9)
    assert isclose(beam.moment_at_point(position=length / 2), 2.5)
    assert isclose(beam.moment_at_point(position=length), 2.5)

    # extracting slope from the beam
    assert False

    # extracting deflection from the beam.
    assert False
