from __future__ import annotations
from typing import Sequence, Optional
import jax
from jax import numpy as jnp
from jax import Array

from algebra.array_utils import ArrayList, check_arr_shape
from utils import _make_pytree


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
    ):
        r"""
        Initialise the flow field.
        :param u_inf: Base flow velocity, [3]
        :param rho: Flow density
        :param relative_motion: If True, the air moves, if false, the plane moves.
        """
        if u_inf.shape != (3,):
            raise ValueError("u_inf must have arr_list_shapes (3,)")
        self.u_inf: Array = u_inf
        self.rho: Array = jnp.array(rho)
        self.u_inf_mag: Array = jnp.linalg.norm(u_inf)
        self.u_inf_dir: Array = u_inf / self.u_inf_mag
        self.q_inf: Array = 0.5 * rho * self.u_inf_mag ** 2

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
        Vectorized version of the __call__ method. This maps over all leading dimensions of x_target.
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

    def to_design_variables(self) -> dict[str, Array]:
        r"""
        Extract the design variables associated with this flow field.
        :return: Dictionary of design variables.
        """
        return {'u_inf': self.u_inf, 'rho': self.rho}

    def from_design_variables(self, design_variables: dict[str, Array]) -> FlowField:
        r"""
        Create a new flow field from a design variables. This allows for design derivatives.
        :param design_variables: Dictionary of design variables.
        :return: New FlowField object.
        """
        return self.__class__(**design_variables,
                              relative_motion=self.relative_motion)

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


@_make_pytree
class Constant(FlowField):
    r"""
    Constant velocity flow field:
    """

    def __call__(self, x: Array, t: Array) -> Array:
        if self.relative_motion:
            return self.u_inf
        else:
            return jnp.zeros(3)


@_make_pytree
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
            self,
            u_inf: Array,
            rho: float | Array,
            relative_motion: bool,
            gust_length: float | Array,
            gust_amplitude: float | Array,
            gust_travel_direction: Optional[Array] = None,
            gust_amplitude_direction: Array = jnp.array((0.0, 0.0, 1.0)),
            gust_x0: Array = jnp.zeros(3),
    ):
        super().__init__(u_inf, rho, relative_motion)

        # base gust parameters
        self.gust_amplitude: Array = jnp.array(gust_amplitude)
        self.gust_length: Array = jnp.array(gust_length)

        # direction of travel for the gust - use background flow direction as default
        # even for a gust frozen in place, this defines the orientation of the ridge
        self.gust_travel_direction: Array = (
            gust_travel_direction
            if gust_travel_direction is not None
            else self.u_inf_dir
        )
        check_arr_shape(self.gust_travel_direction, (3,), "gust_travel_direction")
        self.gust_travel_direction /= jnp.linalg.norm(self.gust_travel_direction)

        # lateral direction of the gust (direction in which the gust acts), default is in Z
        check_arr_shape(gust_amplitude_direction, (3,), "gust_amplitude")
        self.gust_amplitude_direction: Array = (
                gust_amplitude_direction / jnp.linalg.norm(gust_amplitude_direction)
        )

        # base coordinate at the start of the gust at t=0
        self.gust_x0: Array = gust_x0
        check_arr_shape(self.gust_x0, (3,), "gust_x0")

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

    def to_design_variables(self) -> dict[str, Array]:
        r"""
        Extract the design variables associated with this flow field.
        :return: Dictionary of design variables.
        """
        return {'u_inf': self.u_inf, 'rho': self.rho, 'gust_amplitude': self.gust_amplitude,
                'gust_length': self.gust_length}

    def from_design_variables(self, design_variables: dict[str, Array]) -> OneMinusCosine:
        r"""
        Create a new flow field from a design variables. This allows for design derivatives.
        :param design_variables: Dictionary of design variables.
        :return: New FlowField object.
        """
        return OneMinusCosine(**design_variables,
                              relative_motion=self.relative_motion, gust_travel_direction=self.gust_travel_direction,
                              gust_amplitude_direction=self.gust_amplitude_direction, gust_x0=self.gust_x0)

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
