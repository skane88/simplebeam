"""
Basic Beam element class.
"""

from numbers import Number
from typing import Union, Optional

from sympy.physics.continuum_mechanics.beam import Beam as symBeam  # type: ignore

from simplebeam.loads import Load


class Beam:
    """
    A basic Beam element class.
    """

    _loads: list[Load]

    def __init__(
        self,
        *,
        elastic_modulus: Number,
        second_moment: Number,
        length: Number,
        restraints=None,
        loads: Union[list[Load], Load] = None
    ):
        """

        :param elastic_modulus: The elastic modulus.
        :param second_moment: The second moment of inertia.
        :param length: The length of the beam.
        :param restraints: Any restraints applied to the beam.
        :param loads: Any loads applied to the beam.
        """

        self.elastic_modulus = elastic_modulus
        self.second_moment = second_moment
        self.length = length
        self.restraints = restraints

        self._loads = []
        self.add_load(load=loads)

        self._symbeam = symBeam(
            length=self.length,
            elastic_modulus=self.elastic_modulus,
            second_moment=self.second_moment,
        )
        self._solved = False

    @property
    def solved(self):
        """
        Is the beam solved?
        """

        return self._solved

    @property
    def elastic_modulus(self):
        """
        The elastic modulus of the beam.
        """

        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, elastic_modulus):

        self._solved = False
        self._elastic_modulus = elastic_modulus

    @property
    def second_moment(self):
        """
        The second moment of inertia of the beam.
        """

        return self._second_moment

    @second_moment.setter
    def second_moment(self, second_moment):

        self._solved = False
        self._second_moment = second_moment

    @property
    def length(self):
        """
        The length of the beam.
        """

        return self._length

    @length.setter
    def length(self, length):

        self._solved = False
        self._length = length

    @property
    def restraints(self):
        """
        The restraints for the beam.
        """

        return self._restraints

    @restraints.setter
    def restraints(self, restraints):

        self._solved = False
        self._restraints = restraints

    @property
    def loads(self) -> list[Load]:
        """
        The loads on the beam.
        """

        return self._loads

    @loads.setter
    def loads(self, loads: Optional[list[Load]] = None):
        self._solved = False

        self._loads = [] if loads is None else loads

    def add_load(self, load: Union[list[Load], Load] = None):
        """
        Add a load onto the beam.

        :param load: The load to add.
        """

        self._solved = False

        if load is None:
            return

        if isinstance(load, list):

            for individual_load in load:
                self._loads.append(individual_load)

        else:
            self._loads.append(load)
