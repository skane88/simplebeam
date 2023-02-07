"""
File contains a custom exception hierarchy.
"""


class SimpleBeamError(Exception):
    """
    Parent class in an exception hierarchy for the simplebeam project.
    """


class BeamError(SimpleBeamError):
    """
    Parent class for errors in the Beam classes.
    """


class BeamNotSolvedError(BeamError):
    """
    Error to raise if the beam has to be solved before requesting a property.
    """


class PointNotOnBeamError(BeamError):
    """
    Error to raise if a result is requested at a point that is not on the beam.
    """


class LoadError(SimpleBeamError):
    """
    Parent class for errors in the Load classes.
    """


class LoadPositionError(LoadError):
    """
    Class for errors related to the position of the load.
    """


class RestraintError(SimpleBeamError):
    """
    Parent class for errors in the Restraint classes.
    """


class RestraintPositionError(RestraintError):
    """
    Class for errors related to the position of the restraint.
    """
