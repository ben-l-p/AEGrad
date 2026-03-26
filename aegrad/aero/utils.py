from __future__ import annotations
from typing import Sequence, Optional, Callable

import jax
from jax import numpy as jnp, Array
from jax import vmap
from jax.lax import cond

from aegrad.algebra.array_utils import neighbour_average, ArrayList
from algebra.array_utils import split_to_vertex
from constants import EPSILON, R_CUTOFF

type KernelFunction = Callable[[Array, Array], Array]


def make_rectangular_grid(
    m: int, n: int, chord: Array | float, ea: Array | float
) -> Array:
    r"""
    Create a rectangular grid of points in the yz-plane
    :param m: Number of panels in the chordwise direction
    :param n: Number of panels in the spanwise direction
    :param chord: Total chord of the surface
    :param ea: Elastic axis location as fraction of chord
    :return: Array of arr_list_shapes [zeta_m, zeta_n, 3] representing local grid points in 3D space
    """

    grid = jnp.zeros((m + 1, n + 1, 3))
    return grid.at[..., 0].set((jnp.linspace(0.0, chord, m + 1) - ea * chord)[:, None])


def compute_surf_c(zeta: Array) -> Array:
    r"""
    Compute the colocation points for a given grid of points on a single surface.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3]
    :return: Colocation points [zeta_m-1, zeta_n-1, 3]
    """
    return neighbour_average(zeta, axes=(0, 1))


def compute_surf_nc(zeta: Array) -> Array:
    r"""
    Compute the varphi vectors for a given grid of points on a single surface. These have length equal to the area of
    each panel.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3]
    :return: Normal vectors [zeta_m-1, zeta_n-1, 3]
    """
    diag1 = zeta[1:, 1:, :] - zeta[:-1, :-1, :]  # [n_sx, n_cy, 3]
    diag2 = zeta[1:, :-1, :] - zeta[:-1, 1:, :]
    return jnp.cross(diag1, diag2)


def compute_c(zetas: ArrayList) -> ArrayList:
    r"""
    Compute the colocation points for a list of surface grids.
    :param zetas: Grids of points, [n_surf][zeta_m, zeta_n, 3]
    :return: Colocation points [n_surf][zeta_m-1, zeta_n-1, 3]
    """
    return ArrayList([compute_surf_c(zeta) for zeta in zetas])


def compute_nc(zetas: ArrayList) -> ArrayList:
    r"""
    Compute the varphi vectors for a list of surface grids.
    :param zetas: Grids of points, [n_surf][zeta_m, zeta_n, 3]
    :return: Normal vectors [n_surf][zeta_m-1, zeta_n-1, 3]
    """
    return ArrayList([compute_surf_nc(zeta) for zeta in zetas])


def calculate_steady_forcing(
    zeta_bs: ArrayList,
    zeta_dot_bs: Optional[ArrayList],
    gamma_bs: ArrayList,
    gamma_ws: ArrayList,
    rho: Array,
    v_func: Callable[[Array], Array],
    v_inputs: Optional[ArrayList],
) -> ArrayList:
    r"""
    Calculate steady aerodynamic forcing for all surfaces at specified time step
    :param zeta_bs: Bound grids, [n_surf][zeta_m, zeta_n, 3]
    :param zeta_dot_bs: Bound grids velocities, [n_surf][zeta_m, zeta_n, 3]
    :param gamma_bs: Bound grid circulation, [n_surf][gamma_m, gamma_n]
    :param gamma_ws: Bound grid circulation, [n_surf][gamma_m, gamma_n]
    :param rho: Flowfield density
    :param v_func: Total velocity as a function of coordinate
    :param v_inputs: Additive inputs for total velocity on bound grid vertex
    """

    f_steady = ArrayList([])

    if zeta_dot_bs is None:
        zeta_dot_bs = [None] * len(zeta_bs)

    if v_inputs is None:
        v_inputs = [None] * len(zeta_bs)

    for zeta_b, zeta_dot_b, gamma_b, gamma_w, v_input in zip(
        zeta_bs, zeta_dot_bs, gamma_bs, gamma_ws, v_inputs
    ):
        # compute midpoints
        mp_chordwise = neighbour_average(zeta_b, axes=0)  # [gamma_m, gamma_n+1, 3]
        mp_spanwise = neighbour_average(zeta_b, axes=1)  # [gamma_m+1, gamma_n, 3]

        mp_dot_chordwise = neighbour_average(
            zeta_dot_b, axes=0
        )  # [gamma_m, gamma_n+1, 3]
        mp_dot_spanwise = neighbour_average(
            zeta_dot_b, axes=1
        )  # [gamma_m+1, gamma_n, 3]

        # relative flow velocities at midpoints
        v_rel_chordwise = (
            v_func(mp_chordwise) - mp_dot_chordwise
        )  # [gamma_m, gamma_n+1, 3]
        v_rel_spanwise = (
            v_func(mp_spanwise) - mp_dot_spanwise
        )  # [gamma_m+1, gamma_n, 3]

        # add any input_ velocities
        if v_input is not None:
            v_rel_chordwise += neighbour_average(v_input, axes=0)
            v_rel_spanwise += neighbour_average(v_input, axes=1)

        # equivelant strengths of filaments
        gamma_chordwise = jnp.zeros(
            v_rel_chordwise.shape[:-1]
        )  # [gamma_m, gamma_n+1, 3]
        gamma_chordwise = gamma_chordwise.at[:, :-1].set(gamma_b)
        gamma_chordwise = gamma_chordwise.at[:, 1:].add(-gamma_b)
        gamma_spanwise = jnp.zeros(v_rel_spanwise.shape[:-1])  # [gamma_m+1, gamma_n, 3]
        gamma_spanwise = gamma_spanwise.at[:-1, :].set(-gamma_b)
        gamma_spanwise = gamma_spanwise.at[1:, :].add(gamma_b)

        # add first wake gamma
        if gamma_w.shape[0] > 0:
            gamma_spanwise = gamma_spanwise.at[-1, :].add(-gamma_w[0, :])

        # filament vectors (from zeta_b_fil, which may differ from the midpoint geometry)
        r_chordwise = zeta_b[1:, :, :] - zeta_b[:-1, :, :]  # [gamma_m, gamma_n+1, 3]
        r_spanwise = zeta_b[:, 1:, :] - zeta_b[:, :-1, :]  # [gamma_m+1, gamma_n, 3]

        # forces from each set of filaments
        f_chordwise = rho * jnp.einsum(
            "ij,ijk->ijk",
            gamma_chordwise,
            jnp.cross(v_rel_chordwise, r_chordwise),
        )  # [gamma_m, gamma_n+1, 3]
        f_spanwise = rho * jnp.einsum(
            "ij,ijk->ijk", gamma_spanwise, jnp.cross(v_rel_spanwise, r_spanwise)
        )  # [gamma_m+1, gamma_n, 3]

        f_steady.append(
            split_to_vertex(f_chordwise, 0) + split_to_vertex(f_spanwise, 1)
        )  # [gamma_m+1, gamma_n+1, 3]
    return f_steady


def propagate_surf_wake(
    gamma_b_n: Array,
    gamma_w_n: Array,
    zeta_b_np1: Array,
    zeta_w_n: Array,
    delta_w: Optional[Array],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: bool,
) -> tuple[Optional[Array], Array]:
    r"""
    Convect the wake at some given velocity for a single surface. This step includes convection from the trailing edge and culling the
    downstream data.
    :param gamma_b_n: Bound circulation at time varphi, [m, varphi]
    :param gamma_w_n: Wake circulation at time varphi, [m_star, varphi]
    :param zeta_b_np1: Bound grid at time varphi+1, [zeta_m, zeta_n, 3]
    :param zeta_w_n: Wake grid at time varphi, [zeta_star_m, zeta_n, 3]
    :param delta_w: Desired wake discretisation, [zeta_star_m, 3] or None for uniform
    :param v_func: Function that computes the velocity, [3] -> [3]
    :param dt: Time step
    :param frozen_wake: If true, the grid stays constant with time, useful in the linearised case
    :return: New wake grid and circulation, [zeta_star_m, zeta_n, 3], [zeta_m_star, zeta_n]
    """

    # trailing edge positions and circulations
    zeta_te = zeta_b_np1[-1, ...]  # [zeta_n, 3]
    gamma_te = gamma_b_n[-1, ...]  # [gamma_n]

    # variable wake discretisation also depends on the final element
    if delta_w is not None:
        zeta_base = zeta_w_n  # [zeta_w_m, zeta_n, 3]
        gamma_base = gamma_w_n  # [gamma_w_m, gamma_n]
    else:
        zeta_base = zeta_w_n[:-1, ...]  # [zeta_w_m - 1, zeta_n, 3]
        gamma_base = gamma_w_n[:-1, ...]  # [gamma_w_m - 1, gamma_n]

    # values at t=varphi+1 before rediscretisation
    gamma_w_np1 = jnp.concatenate(
        (gamma_te[None, ...], gamma_base), axis=0
    )  # [gamma_w_m+1 | gamma_w_m, gamma_n]

    # if the wake is free, this should be embedded here
    v = v_func(zeta_base)  # [zeta_w_m | zeta_w_m-1, zeta_n, 3]

    # wake coordinates at t=varphi+1 before rediscretisation
    zeta_w_np1 = jnp.concatenate(
        (zeta_te[None, :, :], zeta_base + dt * v), axis=0
    )  # [zeta_w_m+1 | zeta_w_m, zeta_n, 3]

    if delta_w is not None:
        # streamline coordinates before rediscretisation
        s_zeta_w = jnp.concatenate(
            (
                jnp.zeros((1, zeta_te.shape[0])),  # [1, zeta_n]
                jnp.cumsum(
                    jnp.linalg.norm(
                        zeta_w_np1[1:, ...] - zeta_w_np1[:-1, ...], axis=-1
                    ),  # [zeta_w_m+1, zeta_n]
                    axis=0,
                ),  # [zeta_w_m, zeta_n]
            ),
            axis=0,
        )  # distance along each wake filament for each point [zeta_w_m + 1, zeta_n]

        # consider gamma to be at midpoints of zeta
        s_gamma_w = neighbour_average(
            s_zeta_w, axes=(0, 1)
        )  # [gamma_w_m + 1, gamma_w_n]

        # vertex coordinates along desired discretized streamline, [m_star + 1]
        s_zeta_w_redisc = jnp.concatenate((jnp.zeros(1), jnp.cumsum(delta_w)))

        # midpoint coordinates along desired discretized streamline, [m_star]
        s_gamma_w_redisc = neighbour_average(s_zeta_w_redisc, axes=(0,))

        # rediscretise coordinates onto desired grid
        zeta_w_np1 = vmap(
            vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=1),
            in_axes=(None, None, 1),
            out_axes=2,
        )(
            s_zeta_w_redisc, s_zeta_w.T, jnp.transpose(zeta_w_np1, (1, 2, 0))
        )  # [zeta_w_m, zeta_n, 3]

        # rediscretise gamma onto desired grid
        gamma_w_np1 = vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=1)(
            s_gamma_w_redisc, s_gamma_w.T, gamma_w_np1.T
        )  # [zeta_w_m, zeta_n, 3]

    if frozen_wake:
        return None, gamma_w_np1
    else:
        return zeta_w_np1, gamma_w_np1


def propagate_wake(
    gamma_b_n: ArrayList,
    gamma_w_n: ArrayList,
    zeta_b_np1: ArrayList,
    zeta_w_n: ArrayList,
    delta_w: Sequence[Optional[Array]],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: bool,
) -> tuple[ArrayList, ArrayList]:
    r"""
    Convect the wake at some given velocity for all surfaces. This step includes convection from the trailing edge and
    culling the downstream data.
    :param gamma_b_n: Bound circulation at time varphi, [n_surf][m, varphi]
    :param gamma_w_n: Wake circulation at time varphi, [n_surf][m_star, varphi]
    :param zeta_b_np1: Bound grid at time varphi+1, [n_surf][zeta_m, zeta_n, 3]
    :param zeta_w_n: Wake grid at time varphi, [n_surf][zeta_star_m, zeta_n, 3]
    :param delta_w: Desired wake discretisation, [n_surf][zeta_star_m, 3] or None for uniform
    :param v_func: Function that computes the velocity, [3] -> [3]
    :param dt: Time step
    :param frozen_wake: If true, the grid stays constant with time, useful in the linearised case
    :return: New wake grid and circulation, [n_surf][zeta_star_m, zeta_n, 3], [n_surf][m_star, varphi]
    """

    n_surf = len(gamma_b_n)
    zeta_w_np1 = ArrayList([])
    gamma_w_np1 = ArrayList([])

    for i_surf in range(n_surf):
        surf_zeta_w, surf_gamma_w = propagate_surf_wake(
            gamma_b_n[i_surf],
            gamma_w_n[i_surf],
            zeta_b_np1[i_surf],
            zeta_w_n[i_surf],
            delta_w[i_surf],
            v_func,
            dt,
            frozen_wake,
        )
        zeta_w_np1.append(surf_zeta_w)
        gamma_w_np1.append(surf_gamma_w)
    return zeta_w_np1, gamma_w_np1


def biot_savart(x: Array, y: Array) -> Array:
    r"""
    Basic Biot-Savart kernel without any smoothing or cutoff.
    :param x: Target point, [3]
    :param y: Filament endpoints, [2, 3]
    :return: Influence at target point, [3]
    """
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = r1 / jnp.linalg.norm(r1) - r2 / jnp.linalg.norm(r2)
    return r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2) * 4.0 * jnp.pi) * jnp.dot(r0, diff_r)


@jax.custom_jvp
def make_unit_epsilon(r: Array) -> Array:
    r"""
    Differentiable function to obtain a smoothed unit vector. As r -> 0, the output approaches zero instead of being
    undefined.
    :param r: Vector to be normalized, [3]
    :return: Unit vector, [3]
    """
    return r / jnp.sqrt(jnp.sum(r**2) + EPSILON**2)


@make_unit_epsilon.defjvp
def smooth_unit_vector_jvp(primals, tangents):
    r"""
    Custom JVP rule for the smoothed unit vector function.
    """
    (r,) = primals
    (r_dot,) = tangents
    r_norm2 = jnp.sum(r**2)
    r_norm = jnp.sqrt(r_norm2)

    jvp = jax.lax.select(
        r_norm > R_CUTOFF,
        r_dot / (r_norm + EPSILON)
        - jnp.outer(r, r)
        @ r_dot
        / (jnp.sqrt(r_norm2 + EPSILON**2) * (r_norm + EPSILON) ** 2),
        jnp.zeros(3),
    )

    y = r / (jnp.sqrt(r_norm2 + EPSILON**2))
    return y, jvp


def biot_savart_epsilon(x: Array, y: Array) -> Array:
    r"""
    Biot-Savart kernel with epsilon term added to remove singularity.
    :param x: Target point, [3]
    :param y: Filament endpoints, [2, 3]
    :return: Influence at target point, [3]
    """
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = make_unit_epsilon(r1) - make_unit_epsilon(r2)
    r1_x_r2_unit = r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2) + EPSILON)
    return r1_x_r2_unit / (4.0 * jnp.pi) * jnp.dot(r0, diff_r)


def biot_savart_cutoff(x: Array, y: Array) -> Array:
    r"""
    Biot-Savart kernel with truncation radius to remove singularity.
    :param x: Target point, [3]
    :param y: Filament endpoints, [2, 3]
    :return: Influence at target point, [3]
    """
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]

    sm = jnp.inner(r0, r1) / jnp.inner(r0, y[1, :] - y[0, :])
    m = y[0, :] + sm * (y[1, :] - y[0, :])
    r = jnp.linalg.norm(x - m)  # radial distance

    def _kernel_value() -> Array:
        # Compute the standard Biot-Savart kernel, called only if r > R_CUTOFF
        r1_x_r2 = jnp.cross(r1, r2)
        r1_x_r2_unit2 = r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2))
        diff_r = make_unit_epsilon(r1) - make_unit_epsilon(r2)
        return r1_x_r2_unit2 / (4.0 * jnp.pi) * jnp.dot(r0, diff_r)

    return cond((r > R_CUTOFF), _kernel_value, lambda: jnp.zeros(3))


def mirror_grid(zeta: Array, mirror_point: Array, mirror_normal: Array) -> Array:
    """
    Mirror a grid of points across a plane defined by a point and a varphi vector.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3].
    :param mirror_point: Point in mirror plane, [3].
    :param mirror_normal: Normal vector of mirror plane, [3]. Should be normalized.
    :return: Mirrored grid of points, [zeta_m, zeta_n, 3].
    """
    diff = zeta - mirror_point[None, None, :]  # [zeta_m, zeta_n, 3]
    diff_n = jnp.einsum("ijk,k->ij", diff, mirror_normal)  # [zeta_m, zeta_n]
    return (
        zeta - 2.0 * diff_n[:, :, None] * mirror_normal[None, None, :]
    )  # [zeta_m, zeta_n, 3]
