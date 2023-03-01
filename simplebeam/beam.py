"""
Basic Beam element class.
"""
import math
from enum import Enum
from numbers import Number

import numpy as np
from sympy import Expr, Symbol, lambdify, symbols  # type: ignore
from sympy.physics.continuum_mechanics.beam import Beam as SymBeam  # type: ignore

from simplebeam.exceptions import (
    BeamNotSolvedError,
    LoadPositionError,
    PointNotOnBeamError,
    RestraintPositionError,
)
from simplebeam.loads import Load
from simplebeam.restraints import Restraint, fixed, pin

BEAM_NOT_SOLVED_WARNING = "Beam not yet solved."


class ResultType(Enum):
    """
    Create an Enum type for the different results that can be generated.
    """

    LOAD = "load"
    SHEAR = "shear"
    MOMENT = "moment"
    SLOPE = "slope"
    DEFLECTION = "deflection"


class Beam:
    """
    A basic Beam element class.
    """

    # pylint: disable=R0904

    _loads: list[Load]
    _restraints: list[Restraint]
    _symbeam: SymBeam | None

    def __init__(
        self,
        *,
        elastic_modulus: Number,
        second_moment: Number,
        length: Number,
        restraints: list[Restraint] | Restraint | None = None,
        loads: list[Load] | Load | None = None,
        solve: bool = True,
    ):
        """
        Initialise a Beam object.

        :param elastic_modulus: The elastic modulus.
        :param second_moment: The second moment of inertia.
        :param length: The length of the beam.
        :param restraints: Any restraints applied to the beam.
        :param loads: Any loads applied to the beam. Can be applied later with
            self.add_load.
        :param solve: If possible to solve, do so.
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

        if solve and self.solveable:
            self.solve()

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
    def restraints(self, restraints: list[Restraint] | Restraint | None = None):
        self._solved = False
        self._restraints = []
        self.add_restraint(restraint=restraints)

    def add_restraint(self, *, restraint: list[Restraint] | Restraint | None = None):
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

        :param restraint: The restraint to validate
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
    def loads(self, loads: list[Load] | Load | None = None):
        self._solved = False
        self._loads = []
        self.add_load(load=loads)

    def add_load(self, load: list[Load] | Load | None = None):
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

        :param load: The load to validate
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
                    _restraint_symbol(position=restraint.position, prefix="F"),
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

    @property
    def solveable(self):
        """
        Is the beam solveable in principle?

        NOTE: this does not address beams that are mechanisms, or have other sources of
        singularities. It only checks that the user has provided the types of
        information req'd.
        """

        if self.restraints is None or len(self.restraints) == 0:
            return False

        if self.loads is None or len(self.loads) == 0:
            return False

        return not any(
            [
                self.elastic_modulus is None,
                self.second_moment is None,
                self.length is None,
            ]
        )

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
                    _restraint_symbol(position=restraint.position, prefix="F")
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
            raise BeamNotSolvedError(BEAM_NOT_SOLVED_WARNING)
        if self._symbeam is None:
            raise BeamNotSolvedError(BEAM_NOT_SOLVED_WARNING)

        reactions = self._symbeam.reaction_loads
        ret_val = {}

        for i, rest in enumerate(self.restraints):
            ret_val[i] = {
                "F": reactions[_restraint_symbol(position=rest.position, prefix="F")]
                if rest.dy
                else None
            }

            ret_val[i]["M"] = (
                reactions[_restraint_symbol(position=rest.position, prefix="M")]
                if rest.rz
                else None
            )

        return ret_val

    def _equations(self, result_type: ResultType):
        """
        Return the equations describing the load, shear, moment etc. along the beam
        from the underlying _symbeam object.

        :param result_type: The result type to return.
        :return: a sympy object describing the results along the beam.
        """

        eqn: Expr

        match result_type:
            case result_type.LOAD:
                eqn = self._symbeam.load
            case result_type.SHEAR:
                eqn = self._symbeam.shear_force()
            case result_type.MOMENT:
                eqn = self._symbeam.bending_moment()
            case result_type.SLOPE:
                eqn = self._symbeam.slope()
            case result_type.DEFLECTION:
                eqn = self._symbeam.deflection()

        return eqn

    def _result_at_point(self, position, result_type: ResultType):
        """
        Determine the results at a point along the beam. THis is a helper method for the
        public methods for each result type (shear, moment, slope & deflection).
        """

        if position < 0 or position > self.length:
            raise PointNotOnBeamError(
                f"Requested {result_type} result is not on the beam."
            )

        if not self.solved:
            raise BeamNotSolvedError(BEAM_NOT_SOLVED_WARNING)
        if self._symbeam is None:
            raise BeamNotSolvedError(BEAM_NOT_SOLVED_WARNING)

        if position == self.length:
            position = math.nextafter(position, 0)

        if position == 0:
            position = math.nextafter(position, self.length)

        symbol = self._symbeam.variable

        return self._equations(result_type=result_type).subs(symbol, position).evalf()

    def _load_at_point(self, position):
        """
        Determine the load at a point along the beam.

        NOTE: This is a hidden method because it will
        not accurately show point loads & moments.

        :param position: The position to determine the load at, between 0 and length.
        """

        return self._result_at_point(position=position, result_type=ResultType.LOAD)

    def shear_at_point(self, position):
        """
        Determine the shear at a point along the beam.

        :param position: the point to determine the shear at, between 0 and length.
        """

        return self._result_at_point(position=position, result_type=ResultType.SHEAR)

    def moment_at_point(self, position):
        """
        Determine the moment at a point along the beam.

        :param position: the point to determine the moment at, between 0 and length.
        """

        return self._result_at_point(position=position, result_type=ResultType.MOMENT)

    def slope_at_point(self, position):
        """
        Determine the slope at a point along the beam.

        :param position: the point to determine the slope at, between 0 and length.
        """

        return self._result_at_point(position=position, result_type=ResultType.SLOPE)

    def deflection_at_point(self, position):
        """
        Determine the deflection at a point along the beam.

        :param position: the point to determine the deflection at, between 0 and length.
        """

        return self._result_at_point(
            position=position, result_type=ResultType.DEFLECTION
        )

    def key_positions(self) -> set[float]:
        """
        Return a set containing key points along the length of a beam. These are:

        * The start & end of a beam.
        * The location of any loads or restraints that cause discontinuities in the
            Shear, moment, slope or deflection curves.
        """

        def offset_points(x: float) -> set[float]:
            """
            Helper function to build points to add ot the set of points to return.
            """

            ret: set[float] = set()

            if x == 0:
                ret.add(math.nextafter(0, self.length))

            elif x == self.length:
                ret.add(math.nextafter(self.length, 0))

            else:
                ret.add(math.nextafter(x, 0))
                ret.add(math.nextafter(x, self.length))

            return ret

        # first get the start and end of the beam.
        points: set[float] = {
            math.nextafter(0, self.length),
            math.nextafter(self.length, 0),
        }

        for r in self.restraints:
            points = points | offset_points(r.position)

        for load in self.loads:
            if (x := load.start) is not None:
                points = points | offset_points(x)

            if (x := load.end) is not None:
                points = points | offset_points(x)

        return points

    def _result_curve(
        self,
        result_type: ResultType,
        min_points: int = 11,
        user_points: list[float] | float | None = None,
    ) -> tuple[list[float], list[float]]:
        """
        Create a series of x, y points along a result set.

        :param result_type: The result type to query.
        :param min_points: The minimum no. of points to return.
        :param user_points: Points to keep at user defined locations.
        """

        eq = self._equations(result_type=result_type)

        symbol = self._symbeam.variable  # type: ignore

        x, y = get_points(expr=eq, start=0, end=self.length)

        # next make sure that we have the key positions covered

        x_key = self.key_positions()

        if user_points is not None:
            if isinstance(user_points, float):
                user_points = [user_points]

            for up in user_points:
                x_key.add(up)

        for i in np.linspace(0, self.length, min_points):
            x_key.add(i)

        for xk in x_key:
            if xk not in x:
                x.append(xk)
                y.append(eq.subs(symbol, xk).evalf())

        xy = sorted(zip(x, y), key=lambda x: x[0])
        x, y = (list(p) for p in zip(*xy))

        x, y = clean_points(x_coords=x, y_coords=y, x_to_keep=list(x_key))

        if y[0] != 0:
            x = [0.0] + x
            y = [0.0] + y

        if x[-1] != self.length or y[-1] != 0:
            x = x + [self.length]
            y = y + [0.0]

        return x, y

    def _load_curve(
        self, min_points: int = 11, user_points: list[float] | float | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Generate a list of x, y points that define the load curve.

        NOTE: This is a hidden method because it will
        not accurately show point loads & moments.
        """

        return self._result_curve(
            result_type=ResultType.LOAD, min_points=min_points, user_points=user_points
        )

    def shear_curve(
        self, min_points: int = 11, user_points: list[float] | float | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Generate a list of x, y points that define the shear curve.
        """
        return self._result_curve(
            result_type=ResultType.SHEAR, min_points=min_points, user_points=user_points
        )

    def moment_curve(
        self, min_points: int = 11, user_points: list[float] | float | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Generate a list of x, y points that define the moment curve.
        """
        return self._result_curve(
            result_type=ResultType.MOMENT,
            min_points=min_points,
            user_points=user_points,
        )

    def slope_curve(
        self, min_points: int = 11, user_points: list[float] | float | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Generate a list of x, y points that define the slope curve.
        """
        return self._result_curve(
            result_type=ResultType.SLOPE, min_points=min_points, user_points=user_points
        )

    def deflection_curve(
        self, min_points: int = 11, user_points: list[float] | float | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Generate a list of x, y points that define the deflection curve.
        """
        return self._result_curve(
            result_type=ResultType.DEFLECTION,
            min_points=min_points,
            user_points=user_points,
        )

    def __repr__(self):
        restraints = [r.short_name for r in self.restraints]

        return (
            f"{type(self).__name__} "
            + f"length = {self.length} "
            + f"with restraints={repr(restraints)} "
            + f"and {len(self.loads)} loads. "
            + f"Solved={self.solved}."
        )


def _restraint_symbol(*, position, prefix: str) -> Symbol:
    """
    Returns a variable for the unknown reaction that will occur at a position.

    :param: The position of the unknown.
    :prefix: Nominally "F" for a force and "M" for a moment reaction.
    """

    return symbols(f"{prefix}_" + str(position).replace(".", "_"))


def get_points(expr, start, end, min_depth: int = 6, max_depth: int = 8):
    """
    Evaluates an expression across a range, using a recursive algorithm to try
    and identify discontinuities etc. by checking if 3x points are collinear
    within a tolerance. If not, additional points are added in between until the
    tolerance is met or the maximum recursive depth is reached.

    Based on code from SymPy's plotting functions, simplified down as we do not need
    to handle complex numbers etc.

    See https://github.com/sympy/sympy or
    https://www.sympy.org/en/index.html for original code.

    :param expr: the expression to evaluate.
    :param start: the start of the range to evaluate.
    :param end: the end of the range to evaluate.
    :param min_depth: the minimum depth of the recursive algorithm. Essentially sets
        a minimum level of quality of the approximation to the function.
    :param max_depth: the maximum depth of the recursive algorithm. Ensures the
        algorithm does not run forever. Set a higher number for better quality.
    :returns: A list of x co-ordinates and a list of y co-ordinates.
    """

    x_coords = []
    y_coords = []

    x_sym = symbols("x")

    def func(x_val):
        """
        A closure around the sympy expression that allows it to be evaluated to float
        at a point.
        :param x_val: The x-point at which the expression is evaluated.
        """

        return expr.subs(x_sym, x_val).evalf()

    def sample(p, q, depth):
        """
        Samples recursively if three points are almost collinear.
        For depth < min_depth, points are added irrespective of whether they
        satisfy the collinearity condition or not. The maximum depth
        allowed is max_depth.

        :param p: The first point to sample between.
        :param q: The second point to sample between.
        :param depth: The current depth of the recursive algorithm.
        """
        # Randomly sample to avoid aliasing.
        random = 0.45 + np.random.rand() * 0.1

        xnew = p[0] + random * (q[0] - p[0])

        ynew = func(xnew)
        new_point = np.array([xnew, ynew])

        # Sample if the points are not collinear, or if the depth is less than the
        # minimum depth. We are not using np.linspace to avoid aliasing.
        if depth <= max_depth and (not flat(p, new_point, q) or depth < min_depth):
            sample(p, new_point, depth + 1)
            sample(new_point, q, depth + 1)

        # got rid of the block of code here and in the next elif that handled complex
        # numbers because that should not be an issue in the beam equations.

        else:
            x_coords.append(q[0])
            y_coords.append(q[1])

    f_start = func(start)
    f_end = func(end)
    x_coords.append(start)
    y_coords.append(f_start)
    sample(np.array([start, f_start]), np.array([end, f_end]), 0)

    return x_coords, y_coords


def flat(x, y, z, eps=1e-7):
    """
    Checks whether three points are almost collinear

    :param x: the first point.
    :param y: the second point.
    :param z: the third point.
    :param eps: The tolerance for flatness.
    """

    vector_a = (x - y).astype(np.float64)
    vector_b = (z - y).astype(np.float64)

    dot_product = np.dot(vector_a, vector_b)

    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)

    cos_theta = dot_product / (vector_a_norm * vector_b_norm)

    return abs(cos_theta + 1) < eps


def clean_points(x_coords, y_coords, x_to_keep=None):
    """
    Take a list of points in and clean out any points that form a straight line so that
    only the end of each straight segment is kept.

    :param x_coords: The x-coordinates.
    :param y_coords: The y-coordinates.
    :param x_to_keep: A list of x-co-ordinates that should be kept (if any).
    :return: x_coord, y_coord. Note that the input lists are edited in place.
        If you want to keep the original lists then make sure to deep copy
        the coordinates first.
    """

    if x_to_keep is None:
        x_to_keep = [x_coords[0]]

    x_to_keep = np.array(x_to_keep)

    if len(x_coords) < 3:
        return x_coords, y_coords

    i1, i2, i3 = 0, 1, 2

    cleaned = False

    while not cleaned:
        # now go through the points and clean out any points that are on
        # a straight line.

        p1 = np.array([x_coords[i1], y_coords[i1]])
        p2 = np.array([x_coords[i2], y_coords[i2]])
        p3 = np.array([x_coords[i3], y_coords[i3]])

        keep = np.any(np.isclose(x_to_keep, x_coords[i2]))

        if not keep and flat(x=p1, y=p2, z=p3):
            # if flat, then we can remove the middle point.

            del x_coords[i2]
            del y_coords[i2]
        else:
            # if not, we need to update the 3x indexes to the next 3x points.
            i1 += 1
            i2 += 1
            i3 += 1

        # now we need to check to see if we are now at the end of the list

        if i3 == len(x_coords):
            # if i3 is beyond the end of the list then we have cleaned the whole list.

            cleaned = True

    return x_coords, y_coords


def simple(
    length,
    elastic_modulus=200e9,
    second_moment=1.0,
    loads: Load | list[Load] | None = None,
):
    """
    Helper function to create a simply supported beam.

    Creates a beam with a pin support at each end.

    :param length: the length of the beam to create.
    :param elastic_modulus: the elastic modulus of the beam.
    :param second_moment: the second moment of inertia of the beam.
    :param loads: the loads to apply.
    """

    r1 = pin(0)
    r2 = pin(length)

    return Beam(
        length=length,
        elastic_modulus=elastic_modulus,
        second_moment=second_moment,
        loads=loads,
        restraints=[r1, r2],
    )


def fix_ended(
    length,
    elastic_modulus=200e9,
    second_moment=1.0,
    loads: Load | list[Load] | None = None,
):
    """
    Helper function to create a fixed ended beam.

    Creates a beam with a fixed support at each end.

    :param length: the length of the beam to create.
    :param elastic_modulus: the elastic modulus of the beam.
    :param second_moment: the second moment of inertia of the beam.
    :param loads: the loads to apply.
    """

    r1 = fixed(0)
    r2 = fixed(length)

    return Beam(
        length=length,
        elastic_modulus=elastic_modulus,
        second_moment=second_moment,
        loads=loads,
        restraints=[r1, r2],
    )


def propped_cantilever(
    length,
    elastic_modulus=200e9,
    second_moment=1.0,
    loads: Load | list[Load] | None = None,
    fixed_left: bool = True,
):
    """
    Helper function to create a propped cantilever beam.

    :param length: the length of the beam to create.
    :param elastic_modulus: the elastic modulus of the beam.
    :param second_moment: the second moment of inertia of the beam.
    :param loads: the loads to apply.
    :param fixed_left: Which end is the fixed end?
    """

    if fixed_left:
        r1 = fixed(0)
        r2 = pin(length)
    else:
        r1 = pin(0)
        r2 = fixed(length)

    return Beam(
        length=length,
        elastic_modulus=elastic_modulus,
        second_moment=second_moment,
        loads=loads,
        restraints=[r1, r2],
    )


def cantilever(
    length,
    elastic_modulus=200e9,
    second_moment=1.0,
    loads: Load | list[Load] | None = None,
    fixed_left: bool = True,
):
    """
    Helper function to create a propped cantilever beam.

    :param length: the length of the beam to create.
    :param elastic_modulus: the elastic modulus of the beam.
    :param second_moment: the second moment of inertia of the beam.
    :param loads: the loads to apply.
    :param fixed_left: Which end is the fixed end?
    """

    r1 = fixed(0) if fixed_left else fixed(length)

    return Beam(
        length=length,
        elastic_modulus=elastic_modulus,
        second_moment=second_moment,
        loads=loads,
        restraints=r1,
    )
