"""
Basic Beam element class.
"""

from numbers import Number
from typing import Union, Optional

from sympy import symbols
from sympy.physics.continuum_mechanics.beam import Beam as SymBeam

from simplebeam.loads import Load
from simplebeam.restraints import Restraint
from simplebeam.exceptions import LoadPositionError, RestraintPositionError


class Beam:
    """
    A basic Beam element class.
    """

    _loads: list[Load]
    _restraints: list[Restraint]
    _symbeam: Optional[SymBeam]

    def __init__(
        self,
        *,
        elastic_modulus: Number,
        second_moment: Number,
        length: Number,
        restraints: Union[list[Restraint], Restraint] = None,
        loads: Union[list[Load], Load] = None,
    ):
        """

        :param elastic_modulus: The elastic modulus.
        :param second_moment: The second moment of inertia.
        :param length: The length of the beam.
        :param restraints: Any restraints applied to the beam.
        :param loads: Any loads applied to the beam. Can be applied later with
            self.add_load.
        """

        self._solved = False
        self.elastic_modulus = elastic_modulus
        self.second_moment = second_moment
        self.length = length

        self._restraints = []
        self.add_restraint(restraint=restraints)

        self._loads = []
        self.add_load(load=loads)

        self._boundary_variables = []
        self._symbeam = None

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
    def restraints(self) -> list[Restraint]:
        """
        The restraints for the beam.
        """

        return self._restraints

    @restraints.setter
    def restraints(self, restraints: Union[list[Restraint], Restraint] = None):

        self._solved = False
        self._restraints = []
        self.add_restraint(restraint=restraints)

    def add_restraint(self, *, restraint: Union[list[Restraint], Restraint] = None):
        """
        Add a restraint to the Beam.

        :param restraint: The restraint or a list of restraints to add.
        """

        if restraint is None:
            return

        if isinstance(restraint, list):
            for individual_restraint in restraint:
                self.validate_restraint(individual_restraint)

                self._restraints.append(individual_restraint)

        else:

            self.validate_restraint(restraint)

            self._restraints.append(restraint)

    def validate_restraint(self, restraint: Restraint, raise_exceptions: bool = True):
        """
        Validate a restraint is correct. Currently only checks that it is located on the
        beam but additional checks may be added.

        :param restraint: The restraint to valicate
        :param raise_exceptions: If True, raise an exception on discovering an invalid
            restraint.
        :return: True if valid.
        """

        if restraint.position < 0 or restraint.position > self.length:

            if raise_exceptions:
                raise RestraintPositionError(
                    "Restraint position must be on the beam "
                    + f"(0 < start < {self.length}). "
                    + f"Received position = {restraint.position}"
                )

            return False

        for i, each_restraint in enumerate(self.restraints):

            if restraint.position == each_restraint.position:
                raise RestraintPositionError(
                    "Restraints must be in different locations. "
                    + f"Restraint {restraint} is located at the same position as "
                    + f"restraint {i} ({each_restraint})"
                )

        return True

    @property
    def loads(self) -> list[Load]:
        """
        The loads on the beam.
        """

        return self._loads

    @loads.setter
    def loads(self, loads: Union[list[Load], Load] = None):

        self._solved = False
        self._loads = []
        self.add_load(load=loads)

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

                self.validate_load(individual_load)
                self._loads.append(individual_load)

        else:

            self.validate_load(load)
            self._loads.append(load)

    def validate_load(self, load: Load, raise_exceptions: bool = True):
        """
        Validate a load is correct. Currently only checks that it is located on the
        beam but additional checks may be added.

        :param load: The load to valicate
        :param raise_exceptions: If True, raise an exception on discovering an invalid
            load.
        :return: True if valid.
        """

        if load.start is not None and (load.start < 0 or load.start > self.length):

            if raise_exceptions:
                raise LoadPositionError(
                    "Load start position must be on the beam "
                    + f"(0 < start < {self.length}). Received start = {load.start}"
                )

            return False

        if load.end is not None and load.end < 0:

            if raise_exceptions:
                raise LoadPositionError(
                    "Load end position must be greater than 0. "
                    + f"Received end = {load.end}"
                )

            return False

        return True

    def _build_symbeam(self):
        """
        Takes the data on the beam and builds the underlying SymPy beam.
        """

        beam = SymBeam(
            length=self.length,
            elastic_modulus=self.elastic_modulus,
            second_moment=self.second_moment,
        )

        for load in self.loads:

            beam.apply_load(
                value=load.magnitude, start=load.start, order=load.order, end=load.end
            )

        boundary_variables = []

        for i, restraint in enumerate(self.restraints):

            if restraint.dy:
                beam.bc_deflection.append((restraint.position, 0))

                bv_symbol = f"R{i}"

                beam.apply_load(bv_symbol, restraint.position, order=-1)

                boundary_variables.append(symbols(bv_symbol))

            if restraint.rz:
                beam.bc_slope.append((restraint.position, 0))

                bv_symbol = f"M{i}"

                beam.apply_load(bv_symbol, restraint.position, order=-2)

                boundary_variables.append(symbols(bv_symbol))

        self._boundary_variables = boundary_variables
        self._symbeam = beam

    def solve_beam(self):
        """
        Solve the underlying SymPy beam object.
        """

        if not self.solved:
            # no need to redo the work if this was already successfully solved.
            # does rely on people using the setters rather than the overwriting
            # protected variables.

            self._build_symbeam()  # build the SymPy beam in case there were changes from
            # whatever was last created.

            self._symbeam.solve_for_reaction_loads(*self._boundary_variables)
            self._solved = True
