"""
Basic Beam element class.
"""

from numbers import Number
from typing import Optional, Union

import numpy as np
from sympy import Symbol, symbols, lambdify  # type: ignore
from sympy.physics.continuum_mechanics.beam import Beam as SymBeam  # type: ignore

from simplebeam.exceptions import (
    BeamNotSolvedError,
    LoadPositionError,
    RestraintPositionError,
)
from simplebeam.loads import Load
from simplebeam.restraints import Restraint


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
        The restraints for the beam. These will be sorted left-to-right (from 0.0 to
        self.length).
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

        self._solved = False

        if isinstance(restraint, list):
            for individual_restraint in restraint:
                self.validate_restraint(individual_restraint)
                self._restraints.append(individual_restraint)

        else:

            self.validate_restraint(restraint)
            self._restraints.append(restraint)

        self._restraints.sort(key=lambda x: x.position)

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

        :param load: The Load object to add, or a list of Load objects.
        """

        if load is None:
            return

        self._solved = False

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

        self._restraints.sort(key=lambda x: x.position)

        for load in self.loads:

            beam.apply_load(
                value=load.magnitude, start=load.start, order=load.order, end=load.end
            )

        for restraint in self.restraints:
            if restraint.dy:
                beam.bc_deflection.append((restraint.position, 0))
                beam.apply_load(
                    _restraint_symbol(position=restraint.position, prefix="R"),
                    restraint.position,
                    order=-1,
                )

            if restraint.rz:
                beam.bc_slope.append((restraint.position, 0))
                beam.apply_load(
                    _restraint_symbol(position=restraint.position, prefix="M"),
                    restraint.position,
                    order=-2,
                )

        self._symbeam = beam

    def solve(self):
        """
        Solve the underlying SymPy beam object.
        """

        if self.solved:
            return
            # no need to redo the work if this was already successfully solved.
            # does rely on people using the setters rather than the overwriting
            # protected variables.

        self._build_symbeam()  # build the SymPy beam in case there were changes from
        # whatever was last created.

        unknowns = []

        for restraint in self.restraints:

            if restraint.dy:
                unknowns.append(
                    _restraint_symbol(position=restraint.position, prefix="R")
                )

            if restraint.rz:
                unknowns.append(
                    _restraint_symbol(position=restraint.position, prefix="M")
                )

        self._symbeam.solve_for_reaction_loads(*unknowns)
        self._solved = True

    @property
    def reactions(self) -> dict[int, dict[str, float]]:
        """
        The reactions on the beam. Returned as a dictionary of the form:

        {
            index of load starting from left:   {
                'R': Vertical reaction (if any) or None
                'M': Moment reaction (if any) or None
            }
        }

        """

        if not self.solved:
            raise BeamNotSolvedError("Beam not yet solved")
        if self._symbeam is None:
            raise BeamNotSolvedError("Beam not yet solved")

        reactions = self._symbeam.reaction_loads
        ret_val = {}

        for i, rest in enumerate(self.restraints):

            ret_val[i] = {
                "R": reactions[_restraint_symbol(position=rest.position, prefix="R")]
                if rest.dy
                else None
            }

            ret_val[i]["M"] = (
                reactions[_restraint_symbol(position=rest.position, prefix="M")]
                if rest.rz
                else None
            )

        return ret_val

    def _get_x_points(self, min_points: int = 25, tolerance: float = 1e6) -> np.ndarray:
        """
        Determine the points along the beam required to accurately represent the moment
        and shear forces along the beam. Works by determining a baseline no. of points
        and then adding in points located at or either side of singularities etc.

        :param min_points: The minimum no. of points to return.
        :param tolerance: The distance either side of a singularity to insert a point,
            as a fraction of the total length.
        :return:
        """

        singularity_tolerance = self.length / tolerance

        base_points = np.linspace(0, self.length, min_points)

        for restraint in self.restraints:
            base_points = np.append(
                base_points,
                [
                    restraint.position - singularity_tolerance,
                    restraint.position,
                    restraint.position + singularity_tolerance,
                ],
            )

        for load in self.loads:

            base_points = np.append(
                base_points,
                [
                    load.start - singularity_tolerance,
                    load.start,
                    load.start + singularity_tolerance,
                ],
            )

            if load.end is not None:
                base_points = np.append(
                    base_points,
                    [
                        load.end - singularity_tolerance,
                        load.end,
                        load.end + singularity_tolerance,
                    ],
                )

        points = np.unique(base_points)

        points = np.delete(points, np.where(points < 0))
        points = np.delete(points, np.where(points > self.length))

        return points

    def __repr__(self):

        restraints = [r.short_name for r in self.restraints]

        return (
            f"{type(self).__name__} "
            + f"length = {self.length} "
            + f"with restraints={repr(restraints)} "
            + f"and {len(self.loads)} loads."
        )


def _restraint_symbol(*, position, prefix: str) -> Symbol:
    """
    Returns a variable for the unknown reaction that will occur at a position.

    :param: The position of the unknown.
    :prefix: Nominally "R" for a force and "M" for a moment reaction.
    """

    return symbols(f"{prefix}_" + str(position).replace(".", "_"))


def get_points(expr, start, end, max_depth: int = 12):
    """ Return lists of coordinates for plotting. Depending on the
    `adaptive` option, this function will either use an adaptive algorithm
    or it will uniformly sample the expression over the provided range.
    Returns
    =======
        x: list
            List of x-coordinates
        y: list
            List of y-coordinates
    Explanation
    ===========
    The adaptive sampling is done by recursively checking if three
    points are almost collinear. If they are not collinear, then more
    points are added between those points.
    References
    ==========
    .. [1] Adaptive polygonal approximation of parametric curves,
           Luiz Henrique de Figueiredo.
    """

    x_coords = []
    y_coords = []

    x = symbols("x")

    # f = lambdify([x], expr)

    def flat(x, y, z, eps=1e-3):
        """
        Checks whether three points are almost collinear
        """

        vector_a = (x - y).astype(np.float64)
        vector_b = (z - y).astype(np.float64)

        dot_product = np.dot(vector_a, vector_b)

        vector_a_norm = np.linalg.norm(vector_a)
        vector_b_norm = np.linalg.norm(vector_b)

        cos_theta = dot_product / (vector_a_norm * vector_b_norm)

        return abs(cos_theta + 1) < eps

    def sample(p, q, depth):
        """
        Samples recursively if three points are almost collinear.
        For depth < 6, points are added irrespective of whether they
        satisfy the collinearity condition or not. The maximum depth
        allowed is 12.
        """
        # Randomly sample to avoid aliasing.
        random = 0.45 + np.random.rand() * 0.1

        xnew = p[0] + random * (q[0] - p[0])

        ynew = expr.subs(x, xnew).evalf()
        new_point = np.array([xnew, ynew])

        # Maximum depth
        if depth > max_depth:
            x_coords.append(q[0])
            y_coords.append(q[1])

        # Sample irrespective of whether the line is flat till the
        # depth of 6. We are not using linspace to avoid aliasing.
        elif depth < 6:
            sample(p, new_point, depth + 1)
            sample(new_point, q, depth + 1)

        # Sample ten points if complex values are encountered
        # at both ends. If there is a real value in between, then
        # sample those points further.
        elif p[1] is None and q[1] is None:

            xarray = np.linspace(p[0], q[0], 10)
            yarray = list(map(f, xarray))

            if not all(y is None for y in yarray):
                for i in range(len(yarray) - 1):
                    if not (yarray[i] is None and yarray[i + 1] is None):
                        sample(
                            [xarray[i], yarray[i]],
                            [xarray[i + 1], yarray[i + 1]],
                            depth + 1,
                        )

        # Sample further if one of the end points in None (i.e. a
        # complex value) or the three points are not almost collinear.
        elif (
            p[1] is None
            or q[1] is None
            or new_point[1] is None
            or not flat(p, new_point, q)
        ):
            sample(p, new_point, depth + 1)
            sample(new_point, q, depth + 1)
        else:
            x_coords.append(q[0])
            y_coords.append(q[1])

    f_start = expr.subs(x, start).evalf()
    f_end = expr.subs(x, end).evalf()
    x_coords.append(start)
    y_coords.append(f_start)
    sample(np.array([start, f_start]), np.array([end, f_end]), 0)

    return (x_coords, y_coords)
