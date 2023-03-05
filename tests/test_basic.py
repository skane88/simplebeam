"""
Initial test file.
"""

from math import isclose, nextafter

import simplebeam


def test_beam_initialisation():
    """
    Very basic test of the Beam class to see if it can be initialised.
    """

    beam = simplebeam.Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=5.0,
        restraints=None,
        loads=None,
    )

    assert beam
    assert not beam.solved


def test_add_load():
    """
    Very basic test of the Beam class to see if a load can be added without crashing.
    """

    beam = simplebeam.Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=5.0,
        restraints=None,
        loads=None,
    )

    load = simplebeam.Load(order="point", magnitude=10, start=2.5)
    beam.add_load(load=load)

    assert not beam.solved
    assert beam


def test_add_restraint():
    """
    Very basic test of the Beam class to see if a restraint can be added without
    crashing.
    """

    beam = simplebeam.Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=5.0,
        restraints=None,
        loads=None,
    )

    r1 = simplebeam.Restraint(position=0.0)
    r2 = simplebeam.Restraint(position=5.0)

    beam.add_restraint(restraint=[r1, r2])

    assert not beam.solved
    assert beam


def test_solve():
    """
    Very basic test of the beam to see if it can be solved.

    NOTE: This does not check the correctness of the answers, only that the solution
    works.
    """

    beam = simplebeam.Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=5.0,
        restraints=None,
        loads=None,
    )

    r1 = simplebeam.Restraint(position=0.0)
    r2 = simplebeam.Restraint(position=5.0)

    beam.add_restraint(restraint=[r1, r2])

    l1 = simplebeam.Load(order="point", magnitude=-1, start=2.5)

    beam.add_load(l1)

    assert not beam.solved

    beam.solve()

    assert beam.solved
    assert beam


def test_reactions():
    """
    Basic test that reactions are returned as expected.
    """

    beam = simplebeam.Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=5.0,
        restraints=None,
        loads=None,
    )

    r1 = simplebeam.Restraint(position=0.0)
    r2 = simplebeam.Restraint(position=5.0)

    beam.add_restraint(restraint=[r1, r2])

    l1 = simplebeam.Load(order="point", magnitude=-1, start=2.5)

    beam.add_load(l1)

    beam.solve()

    expected = {0: {"F": 0.5, "M": -0.625}, 1: {"F": 0.5, "M": 0.625}}

    for support in (0, 1):
        for react in ("F", "M"):
            assert isclose(beam.reactions[support][react], expected[support][react])


def test_key_positions():
    """
    Test the function for determining key positions along a beam.
    """

    length = 5.0
    load1 = simplebeam.point(magnitude=5, position=length * 0.4)
    load2 = simplebeam.point(magnitude=5, position=length * 0.75)
    load3 = simplebeam.udl(magnitude=4, start=length * 0.2, end=length * 0.9)

    r1 = simplebeam.pin(0)
    r2 = simplebeam.pin(2.5)

    expected_points = {
        nextafter(0, length),
        nextafter(0.4 * length, 0),
        nextafter(0.4 * length, length),
        nextafter(0.75 * length, 0),
        nextafter(0.75 * length, length),
        nextafter(0.2 * length, 0),
        nextafter(0.2 * length, length),
        nextafter(0.9 * length, 0),
        nextafter(0.9 * length, length),
        nextafter(2.5, 0),
        nextafter(2.5, length),
        nextafter(length, 0),
    }

    E = 200e9
    I = 0.001

    beam = simplebeam.Beam(
        elastic_modulus=E,
        second_moment=I,
        length=length,
        restraints=[r1, r2],
        loads=[load1, load2, load3],
    )

    beam.solve()

    assert expected_points == beam.key_positions()
