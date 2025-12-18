import jax
from jax import numpy as jnp
from jax import Array
from aegrad.algebra.array_utils import ArrayList
from aegrad.utils import make_pytree
from typing import Sequence


class FlowField:
    r"""
    Base class for flow fields.
    :var u_inf: Background flow velocity, [3]
    :var relative_motion: If True, the air moves, if false, the plane moves.
    """

    def __init__(
        self,
        u_inf: Array,
        rho: float | Array,
        relative_motion: bool,
        **kwargs,
    ):
        r"""
        Initialise the flow field.
        :param u_inf: Base flow velocity, [3]
        :param rho: Flow density
        :param relative_motion: If True, the air moves, if false, the plane moves.
        :param kwargs: Additional keyword arguments for specific flow fields.
        """
        if u_inf.shape != (3,):
            raise ValueError("u_inf must have shapes (3,)")
        self.u_inf: Array = u_inf
        self.rho: Array = jnp.array(rho)
        self.u_inf_mag: Array = jnp.linalg.norm(u_inf)
        self.u_inf_dir: Array = u_inf / self.u_inf_mag
        self.q_inf: Array = 0.5 * rho * self.u_inf_mag**2

        # if False, the background flow is time-independent
        self.relative_motion: bool = relative_motion

    def __call__(self, x: Array, t: Array) -> Array:
        """
        Evaluate the flow field at a spatial and temporal coordinate.

        :param x: Spatial coordinates, [3]
        :param t: Time. []
        :return: Flow field values at the specified coordinates, [3]
        """
        raise NotImplementedError("__call__ method must be implemented in subclasses.")

    def vmap_call(self, x: Array, t: Array) -> Array:
        """
        Vectorized version of the __call__ method. This maps over all leading dimensions of x.
        :param x: Spatial coordinates, [..., 3]
        :param t: Time. []
        :return: Flow field values at the specified coordinates, [..., 3]
        """
        n_vmap = x.ndim - 1
        func = self.__call__
        for i_dim in range(n_vmap):
            func = jax.vmap(func, in_axes=(i_dim, None), out_axes=i_dim)
        return func(x, t)

    def surf_vmap_call(self, xs: ArrayList, t: Array) -> ArrayList:
        """
        Vectorized version of the __call__ method over a list of surfaces.
        :param xs: Spatial coordinates, [n_surf][..., 3]
        :param t: Time. []
        :return: Flow field values at the specified coordinates, [n_surf][..., 3]
        """
        return ArrayList([self.vmap_call(x, t) for x in xs])

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Return the names of dynamic attributes for pytree registration.
        :return: Sequence of dynamic attribute names.
        """
        return []

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Return the names of static attributes for pytree registration.
        :return: Sequence of static attribute names.
        """
        return (
            "u_inf",
            "rho",
            "u_inf_mag",
            "u_inf_dir",
            "q_inf",
            "relative_motion",
        )


@make_pytree
class Constant(FlowField):
    r"""
    Constant velocity flow field:
    """

    def __call__(self, x: Array, t: Array) -> Array:
        if self.relative_motion:
            return self.u_inf
        else:
            return jnp.zeros(3)


@make_pytree
class OneMinusCosine(FlowField):
    r"""
    One minus cosine flow field.
    Takes keyword arguments:
    gust_amplitude: Amplitude of the gust,
    gust_length: Length of the gust,
    gust_travel_direction: Direction of the gust as a vector, [3]
    gust_amplitude_direction: Direction of the gust amplitude as a vector, [3]
    gust_x0: Base coordinate at the start of the gust at t=0, [3]
    """

    def __init__(
        self, u_inf: Array, rho: float | Array, relative_motion: bool, **kwargs
    ):
        super().__init__(u_inf, rho, relative_motion, **kwargs)

        # base gust parameters
        try:
            self.gust_amplitude: Array = kwargs["gust_amplitude"]
            self.gust_length: Array = kwargs["gust_length"]
        except KeyError:
            raise ValueError(
                "gust_amplitude and gust_length must be provided as keyword arguments."
            )

        # direction of travel for the gust - use background flow direction as default
        # even for a gust frozen in place, this defines the orientation of the ridge
        self.gust_travel_direction: Array = kwargs.get(
            "gust_travel_direction", self.u_inf
        )
        if self.gust_travel_direction.shape != (3,):
            raise ValueError("gust_travel_direction must have shapes (3,)")
        self.gust_travel_direction /= jnp.linalg.norm(self.gust_travel_direction)

        # lateral direction of the gust (direction in which the gust acts), default is in Z
        self.gust_amplitude_direction: Array = kwargs.get(
            "gust_amplitude_direction", jnp.array((0.0, 0.0, 1.0))
        )
        if self.gust_amplitude_direction.shape != (3,):
            raise ValueError("gust_amplitude_direction must have shapes (3,)")
        self.gust_amplitude_direction /= jnp.linalg.norm(self.gust_amplitude_direction)

        # base coordinate at the start of the gust at t=0
        self.gust_x0: Array = kwargs.get("gust_x0", jnp.zeros(3))
        if self.gust_x0.shape != (3,):
            raise ValueError("gust_x0 must have shapes (3,)")

    def __call__(self, x: Array, t: Array) -> Array:
        """
        Evaluate the one minus cosine flow field at any spatial and temporal coordinate.

        :param x: Spatial coordinates, [3]
        :param t: Time. []
        :return: One minus cosine flow field value, [3]
        """
        rel_x = x - self.gust_x0  # position relative to the gust start
        if self.relative_motion:
            rel_x -= self.u_inf * t  # add relative motion if applicable

        gust_x = jnp.dot(rel_x, self.gust_travel_direction)

        def _one_minus_cos(x_: Array) -> Array:
            return (
                self.gust_amplitude_direction
                * self.gust_amplitude
                * 0.5
                * (1.0 - jnp.cos(jnp.pi * x_ / self.gust_length))
            )

        u = jax.lax.select(
            (gust_x > 0) & (gust_x < 2.0 * self.gust_length),
            _one_minus_cos(gust_x),
            jnp.zeros(3),
        )

        if self.relative_motion:
            u += self.u_inf
        return u

    @staticmethod
    def _static_names() -> Sequence[str]:
        return (
            *FlowField._static_names(),
            "gust_amplitude",
            "gust_length",
            "gust_travel_direction",
            "gust_amplitude_direction",
            "gust_x0",
        )
