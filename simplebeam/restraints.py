"""
File for the Restraint class.
"""

from typing import Union

from simplebeam.exceptions import RestraintError, RestraintPositionError

VALID_CODES = {"F": True, "R": False}
RESTRAINT_TYPE = {
    (True, True): "Fixed",
    (True, False): "Pin",
    (False, False): "Free",
    (False, True): "Guide",
}
RESTRAINT_CODE = {
    (True, True): "FF",
    (True, False): "FR",
    (False, False): "RR",
    (False, True): "RF",
}


class Restraint:
    """
    An object to represent a restraint for the beam.
    """

    def __init__(
        self, position, dy: Union[str, bool] = True, rz: Union[str, bool] = True
    ):
        """
        Constructor for the restraint class.

        :position: The position of the restraint. Must be between 0. and the length of
            the beam that it will be applied to.
        :param dy: Is the beam restrained for translation in the y direction?
            True if fixed, False if free. Alternatively, use "f" or "r"
        :param rz: Is the beam restrained for rotation about the z axis?
            True if fixed, False if free. Alternatively, use "f" or "r"
        """

        if position < 0:
            raise RestraintPositionError(
                f"Restraint position must be >0. Received position={position}"
            )

        if isinstance(dy, str):

            if dy.upper() not in VALID_CODES:
                raise RestraintError(
                    "Expected dy be either True or False, or part of "
                    + f"{repr(VALID_CODES.keys())}. Received dy = {dy}"
                )

            dy = VALID_CODES[dy]

        if isinstance(rz, str):

            if rz.upper() not in VALID_CODES:
                raise RestraintError(
                    "Expected rz be either True or False, or part of "
                    + f"{repr(VALID_CODES.keys())}. Received rz = {dy}"
                )

            rz = VALID_CODES[rz]

        self._dy = dy
        self._rz = rz
        self._position = position

    @property
    def dy(self):
        """
        Is the beam restrained for translation in the y direction?

        :return: True if fixed, False if free.
        """

        return self._dy

    @property
    def rz(self):
        """
        Is the beam restrained for rotation about the z axis?

        :return: True if fixed, False if free.
        """

        return self._rz

    @property
    def restraint_type(self):
        """
        The restraint type ("Fixed", "Pin", "Guide" or "Free")
        """

        return RESTRAINT_TYPE[(self.dy, self.rz)]

    @property
    def restraint_code(self):
        """
        The restraint code ("FF", "RR", "FR" or "RF")
        """

        return RESTRAINT_CODE[(self.dy, self.rz)]

    @property
    def short_name(self):
        """
        A short name for the restraint based on the restraint code + the position. E.g.
        FF@5.0
        """

        return self.restraint_code + "@" + str(self.position)

    @property
    def position(self):
        """
        The position of the restraint.
        """

        return self._position

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"{self.restraint_type} ({self.restraint_code}) "
            + f"at position={repr(self.position)}"
        )


def pin(position) -> Restraint:
    """
    Return a pinned restraint.

    :param position: The position of the restraint. Must be between 0. and the length of
        the beam that it will be applied to.
    """

    return Restraint(position=position, dy=True, rz=False)


def fixed(position) -> Restraint:
    """
    Return a fixed restraint.

    :param position: The position of the restraint. Must be between 0. and the length of
        the beam that it will be applied to.
    """

    return Restraint(position=position)


def guide(position) -> Restraint:
    """
    Return a guide restraint.

    :param position: The position of the restraint. Must be between 0. and the length of
        the beam that it will be applied to.
    """

    return Restraint(position=position, dy=False, rz=True)
