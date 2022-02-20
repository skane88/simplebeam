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

    r1 = pin(position=0)
    r2 = guide(position=length)
    l1 = point(magnitude=1.0, position=length / 2)

    beam = Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=length,
        restraints=[r1, r2],
        loads=l1,
    )

    beam.solve()

    expected_reactions = {0: {"R": -1.0, "M": None}, 1: {"R": None, "M": 2.5}}

    for support in (0, 1):
        for react in ("R", "M"):

            reaction = beam.reactions[support][react]

            if reaction is not None:

                assert isclose(reaction, expected_reactions[support][react])

            else:
                assert reaction == expected_reactions[support][react]
