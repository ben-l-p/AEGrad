from __future__ import annotations
from jax import Array, vmap
from typing import Sequence, Optional, Callable
from jax import numpy as jnp
from aegrad.array_utils import neighbour_average, ArrayList, split_to_vertex

def make_rectangular_grid(m: int, n: int, chord: float, ea: float) -> Array:
    r"""
    Create a rectangular grid of points in the yz-plane
    :param m: Number of panels in the chordwise direction
    :param n: Number of panels in the spanwise direction
    :param chord: Total chord of the surface
    :param ea: Elastic axis location as fraction of chord
    :return: Array of shapes [m+1, n+1, 3] representing grid points in 3D space
    """

    grid = jnp.zeros((m+1, n+1, 3))
    return grid.at[..., 0].set((jnp.linspace(0.0, chord, m + 1) - ea * chord)[:, None])

def get_surf_c(zeta: Array) -> Array:
    r"""
    Compute the colocation points for a given grid of points.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3]
    :return: Colocation points [zeta_m-1, zeta_n-1, 3]
    """
    return neighbour_average(zeta, axes=(0, 1))


def get_surf_nc(zeta: Array) -> Array:
    diag1 = zeta[1:, 1:, :] - zeta[:-1, :-1, :]  # [n_sx, n_cy, 3]
    diag2 = zeta[1:, :-1, :] - zeta[:-1, 1:, :]
    return jnp.cross(diag1, diag2)

def get_c(zetas: ArrayList) -> ArrayList:
    return ArrayList([get_surf_c(zeta) for zeta in zetas])

def get_nc(zetas: ArrayList) -> ArrayList:
    return ArrayList([get_surf_nc(zeta) for zeta in zetas])


def propagate_surf_wake(gamma_b_n: Array,
                        gamma_w_n: Array,
                        zeta_b_np1: Array,
                        zeta_w_n: Array,
                        delta_w: Optional[Array],
                        v_func: Callable[[Array], Array],
                        dt: Array,
                        frozen_wake: bool) -> tuple[Optional[Array], Array]:
    r"""
    Convect the wake at some given velocity for a single surface. This step includes convection from the trailing edge and culling the
    downstream data.
    :param gamma_b_n: Bound circulation at time n, [m, n]
    :param gamma_w_n: Wake circulation at time n, [m_star, n]
    :param zeta_b_np1: Bound grid at time n+1, [zeta_m, zeta_n, 3]
    :param zeta_w_n: Wake grid at time n, [zeta_w_m, zeta_n, 3]
    :param delta_w: Desired wake discretisation, [zeta_w_m, 3] or None for uniform
    :param v_func: Function that computes the velocity, [3] -> [3]
    :param dt: Time step
    :param frozen_wake: If true, the grid stays constant with time, useful in the linearised case
    :return: New wake grid and circulation, [zeta_w_m, zeta_n, 3], [zeta_w_m, zeta_n]
    """
    zeta_te = zeta_b_np1[-1, ...]  # [zeta_n, 3]
    gamma_te = gamma_b_n[-1, ...] # [gamma_n]

    # variable wake discretisation also depends on the final element
    if delta_w is not None:
        zeta_base = zeta_w_n
        gamma_base = gamma_w_n
    else:
        zeta_base = zeta_w_n[:-1, ...]  # [zeta_w_m, zeta_n, 3]
        gamma_base = gamma_w_n[:-1, ...]

    # values we wish to propagate
    zeta_pre = jnp.concatenate(
        (zeta_te[None, ...], zeta_base), axis=0
    ) # [zeta_w_m+1 | zeta_w_m, zeta_n, 3]
    gamma_pre = jnp.concatenate(
        (gamma_te[None, ...], gamma_base), axis=0
    )   # [gamma_w_m+1 | gamma_w_m, gamma_n]

    # if the wake is free, this should be embedded here
    v = v_func(zeta_pre)

    # find the integrated in time version - this will be the final version if no rediscretisation is needed
    zeta_w_new = zeta_pre + dt * v

    if delta_w is not None:
        zeta_pre_redisc = jnp.concatenate((zeta_te[None, :], zeta_w_new), axis=0)  # [zeta_w_m+2, zeta_n, 3]
        gamma_pre_redisc = jnp.concatenate((gamma_te[None, :], gamma_base), axis=0)  # [gamma_w_m+2, gamma_n]

        # if the wake discretisation is variable, we need to rediscretize the wake
        s_zeta = jnp.concatenate(
            (
                jnp.zeros((1, zeta_te.shape[0])),  # [1, zeta_n]
                jnp.cumsum(
                    jnp.linalg.norm(zeta_pre_redisc[1:, ...] - zeta_pre_redisc[:-1, ...], axis=-1), # [zeta_w_m+1, zeta_n]
                    axis=0,
                ),  # [zeta_w_m, zeta_n]
            ),
            axis=0,
        )   # distance along each wake filament for each point [zeta_w_m + 1, zeta_n]

        # consider gamma to be at midpoints of zeta
        s_gamma = neighbour_average(s_zeta, axes=(0, 1)) # [gamma_w_m + 1, gamma_w_n]

        # coordinates along desired discretized streamline, [zeta_w_m]
        s_base = jnp.cumsum(jnp.linalg.norm(delta_w, axis=-1), axis=0)

        zeta_w_np1 = vmap(
            vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=0),
            in_axes=(None, None, 2),
            out_axes=2,
        )(s_base, s_zeta, zeta_pre_redisc)

        gamma_w_np1 = vmap(
            vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=0),
            in_axes=(None, None, 2),
            out_axes=2,
        )(s_base, s_gamma, gamma_pre_redisc)
    else:
        zeta_w_np1 = zeta_w_new
        gamma_w_np1 = gamma_pre

    if frozen_wake:
        return None, gamma_w_np1
    else:
        return zeta_w_np1, gamma_w_np1

def propagate_wake(gamma_b_n: ArrayList,
                    gamma_w_n: ArrayList,
                    zeta_b_np1: ArrayList,
                    zeta_w_n: ArrayList,
                    delta_w: Sequence[Optional[Array]],
                    v_func: Callable[[Array], Array],
                    dt: Array,
                   frozen_wake: bool) -> tuple[ArrayList, ArrayList]:
    r"""
    Convect the wake at some given velocity for all surfaces. This step includes convection from the trailing edge and
    culling the downstream data.
    :param gamma_b_n: Bound circulation at time n, [n_surf][m, n]
    :param gamma_w_n: Wake circulation at time n, [n_surf][m_star, n]
    :param zeta_b_np1: Bound grid at time n+1, [n_surf][zeta_m, zeta_n, 3]
    :param zeta_w_n: Wake grid at time n, [n_surf][zeta_w_m, zeta_n, 3]
    :param delta_w: Desired wake discretisation, [n_surf][zeta_w_m, 3] or None for uniform
    :param v_func: Function that computes the velocity, [3] -> [3]
    :param dt: Time step
    :param frozen_wake: If true, the grid stays constant with time, useful in the linearised case
    :return: New wake grid and circulation, [n_surf][zeta_w_m, zeta_n, 3], [n_surf][zeta_w_m, zeta_n]
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


def steady_forcing(zeta_b: ArrayList,
                             zeta_dot_b: ArrayList,
                             gamma_b: ArrayList,
                             gamma_w: ArrayList,
                             v_func: Callable[[Array], Array],
                             v_input: Optional[ArrayList],
                             rho: Array) -> ArrayList:
    f_steady = ArrayList([])
    for i_surf in range(len(zeta_b)):
       f_steady.append(surf_steady_forcing(zeta_b[i_surf],
                                           zeta_dot_b[i_surf],
                                           gamma_b[i_surf],
                                           gamma_w[i_surf],
                                           v_func,
                                           v_input[i_surf] if v_input is not None else None,
                                           rho))
    return f_steady


def surf_steady_forcing(zeta_b: Array,
                                  zeta_dot_b: Array,
                                  gamma_b: Array,
                                  gamma_w: Array,
                                  v_func: Callable[[Array], Array],
                                  v_input: Optional[Array],
                                  rho: Array) -> Array:

    # compute midpoints
    mp_chordwise = neighbour_average(zeta_b, axes=0)  # [gamma_m, gamma_n+1, 3]
    mp_spanwise = neighbour_average(zeta_b, axes=1)  # [gamma_m+1, gamma_n, 3]

    mp_dot_chordwise = neighbour_average(zeta_dot_b, axes=0)  # [gamma_m, gamma_n+1, 3]
    mp_dot_spanwise = neighbour_average(zeta_dot_b, axes=1)  # [gamma_m+1, gamma_n, 3]

    # relative flow velocities at midpoints
    v_rel_chordwise = v_func(mp_chordwise) - mp_dot_chordwise  # [gamma_m, gamma_n+1, 3]
    v_rel_spanwise = v_func(mp_spanwise) - mp_dot_spanwise  # [gamma_m+1, gamma_n, 3]

    # add any input velocities
    if v_input is not None:
        v_rel_chordwise += neighbour_average(v_input, axes=0)
        v_rel_spanwise += neighbour_average(v_input, axes=1)

    # equivelant strengths of filaments
    gamma_chordwise = jnp.zeros(v_rel_chordwise.shape[:-1])  # [gamma_m, gamma_n+1, 3]
    gamma_chordwise = gamma_chordwise.at[:, :-1].set(gamma_b)
    gamma_chordwise = gamma_chordwise.at[:, 1:].add(-gamma_b)
    gamma_spanwise = jnp.zeros(v_rel_spanwise.shape[:-1]) # [gamma_m+1, gamma_n, 3]
    gamma_spanwise = gamma_spanwise.at[:-1, :].set(-gamma_b)
    gamma_spanwise = gamma_spanwise.at[1:, :].add(gamma_b)

    # add first wake gamma
    if gamma_w.shape[0] > 0:
        gamma_spanwise = gamma_spanwise.at[-1, :].add(-gamma_w[0, :])

    # filement vectors
    r_chordwise = zeta_b[1:, :, :] - zeta_b[:-1, :, :]  # [gamma_m, gamma_n+1, 3]
    r_spanwise = zeta_b[:, 1:, :] - zeta_b[:, :-1, :]  # [gamma_m+1, gamma_n, 3]

    # forces from each set of filaments
    f_chordwise = rho * jnp.einsum('ij,ijk->ijk', gamma_chordwise, jnp.cross(v_rel_chordwise, r_chordwise))    # [gamma_m, gamma_n+1, 3]
    f_spanwise = rho * jnp.einsum('ij,ijk->ijk', gamma_spanwise, jnp.cross(v_rel_spanwise, r_spanwise))  # [gamma_m+1, gamma_n, 3]

    return split_to_vertex(f_chordwise, 0) + split_to_vertex(f_spanwise, 1) # [gamma_m+1, gamma_n+1, 3]
