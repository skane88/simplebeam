"""
Define a class hierarchy for loads.
"""

from simplebeam.exceptions import LoadPositionError

# load orders based on McCauley's method. Used to set the order of the singularity
# functions used.
ORDERS = {
    "moment": -2,
    "point": -1,
    "constant": 0,
    "udl": 0,
    "ramp": 1,
    "parabolic ramp": 2,
}
LOAD_TYPES_BY_ORDER = {
    -2: "moment",
    -1: "point",
    0: "udl",
    1: "ramp",
    2: "parabolic ramp",
}


class Load:
    """
    Parent class for loads.
    """

    def __init__(self, *, order, magnitude, start=None, end=None):
        """
        The

        :param order: The order of the singularity function to use. May be an integer
            between -2 and 2 or one of the following strings:

            "moment"
            "point"
            "constant"
            "udl"
            "ramp"
            "parabolic ramp"

        :param magnitude: The magnitude of the load.
        :param start: The starting location of the load. If None, it is assumed to be
            at the start of the beam.
        :param end: An optional end location. Not required for moment or point loads.
            If None, it is assumed the load extends to the end of the beam (for
            distributed loads).
        """

        self.order = order
        self._magnitude = magnitude

        if start is not None and start < 0:
            raise LoadPositionError(
                f"Load start position must be > 0. Received start = {start}"
            )

        if start is not None and end is not None and end < start:
            raise LoadPositionError(
                f"Expected end to be > start. Received start = {start} > end = {end}"
            )

        self._start = start
        self._end = end

    @property
    def order(self):
        """
        The order of the singularity function used.
        """

        return self._order

    @order.setter
    def order(self, order):

        self._order = ORDERS[order.lower()] if isinstance(order, str) else order

    @property
    def magnitude(self):
        """
        The magnitude of the load.
        """

        return self._magnitude

    @property
    def start(self):
        """
        The starting position of the load.
        """

        return self._start

    @property
    def end(self):
        """
        The ending position of the load.

        For moment and point loads this will return None.
        """

        return self._end

    @property
    def load_type(self):
        """
        Get a string describing the load type.
        """

        return LOAD_TYPES_BY_ORDER[self.order]

    def __repr__(self):
        return (
            f"{type(self).__name__} "
            + f"of type {self.load_type} "
            + f"with magnitude={repr(self.magnitude)} "
            + f"between position={repr(self.start)} and ={repr(self.end)}"
        )
