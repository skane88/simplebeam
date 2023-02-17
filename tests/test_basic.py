"""
Initial test file.
"""

from math import isclose

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
