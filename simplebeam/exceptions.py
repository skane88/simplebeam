"""
File contains a custom exception heirarchy.
"""


class SimpleBeamError(Exception):
    """
    Parent class in an exception heirarchy for the simplebeam project.
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
