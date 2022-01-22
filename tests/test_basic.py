"""
Initial test file.
"""

import simplebeam


def test_beam_initialisation():
    """
    Very basic test of the Beam class to see if it can be initialised.
    """

    beam = simplebeam.Beam(
        elastic_modulus=200e9,
        second_moment=1.0,
        length=5.0,
        restraints=["FR", "FR"],
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
        restraints=["FR", "FR"],
        loads=None,
    )

    load = simplebeam.Load(order="point", magnitude=10, start=2.5)
    beam.add_load(load=load)

    assert not beam.solved
