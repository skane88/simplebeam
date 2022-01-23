"""
File for the Restraint class.
"""

from typing import Union

from simplebeam.exceptions import RestraintError, RestraintPositionError

VALID_CODES = {"f": True, "r": False}
RESTRAINT_TYPE = {(True, True): "fixed", (True, False): "pin", (False, False): "free"}
RESTRAINT_CODE = {(True, True): "ff", (True, False): "fr", (False, False): "rr"}


class Restraint:
    """
    An object to represent a restraint for the beam.
    """

    def __init__(
        self, position, dy: Union[str, bool] = True, rz: Union[str, bool] = True
    ):
        """
        Constructor for the restraint class.

        :position: The position of the restraint.
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

            if dy.lower() not in VALID_CODES:
                raise RestraintError(
                    "Expected dy be either True or False, or part of "
                    + f"{repr(VALID_CODES.keys())}. Received dy = {dy}"
                )

            dy = VALID_CODES[dy]

        if isinstance(rz, str):

            if rz.lower() not in VALID_CODES:
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
        The restraint type ("fixed", "pin" or "free")
        """

        return RESTRAINT_TYPE[(self.dy, self.rz)]

    @property
    def restraint_code(self):
        """
        The restraint code ("ff", "fr" or "rr")
        """

        return RESTRAINT_CODE[(self.dy, self.rz)]

    @property
    def position(self):
        """
        The position of the restraint.
        """

        return self._position

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"{self.restraint_type} "
            + f"at position={repr(self.position)}"
        )
