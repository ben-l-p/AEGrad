from __future__ import annotations
from typing import Sequence, Optional, Callable, overload, Literal

import jax
from jax import numpy as jnp, Array
from jax import vmap
from jax.lax import cond

from aegrad.algebra.array_utils import neighbour_average, ArrayList
from aegrad.algebra.array_utils import split_to_vertex
from aegrad.utils.constants import EPSILON, R_CUTOFF
from aegrad.algebra.base import finite_difference
from aegrad.algebra.so3 import exp_so3
from aegrad.utils.utils import index_to_arr

type KernelFunction = Callable[[Array, Array], Array]


def make_rectangular_grid(
    m: int, n: int, chord: Array | float, ea: Array | float
) -> Array:
    r"""
    Create a rectangular aerodynamic grid.
    :param m: Number of panels in the chordwise direction.
    :param n: Number of panels in the spanwise direction.
    :param chord: Surface chord length.
    :param ea: Elastic axis location as fraction of chord.
    :return: Local grid points for planar wing, [zeta_m, zeta_n, 3].
    """

    grid = jnp.zeros((m + 1, n + 1, 3))
    return grid.at[..., 0].set((jnp.linspace(0.0, chord, m + 1) - ea * chord)[:, None])


def add_control_surface(
    grid: Array,
    angle: Array,
    m_slice: Array | Sequence[int] | slice,
    n_slice: Array | Sequence[int] | slice,
    hinge_axis: Array = jnp.array((0.0, 1.0, 0.0)),
) -> Array:
    r"""
    Add a control surface to a panel grid.
    :param grid: Grid without deflection of this surface, [zeta_m, zeta_n, 3].
    :param angle: Angle in radians through which the control surface will be deflected.
    :param m_slice: Slice of chordwise strips to include in the control surface.
    :param n_slice: Slice of spanwise strips to include in the control surface.
    :param hinge_axis: Axis of the hinge surface in the local frame, [3].
    :return: Deflected aerodynamic grid, [zeta_m, zeta_n, 3].
    """

    m_slice_arr: Array = index_to_arr(index=m_slice, n_entries=grid.shape[0])
    n_slice_arr: Array = index_to_arr(index=n_slice, n_entries=grid.shape[1])

    # grid for deflected surfaces
    grid_out = grid

    def inner_func(n_idx: Array) -> Array:
        hinge_point = grid[m_slice_arr[0], n_idx, :]  # [3]

        crv = hinge_axis * angle  # cartesian rotation vector for surface, [3].
        rmat = exp_so3(crv)  # rotation matrix for rotating surface

        # transform coordinates to rotate control surface
        return (
            jnp.einsum(
                "ij,hj->hi",
                rmat,
                (grid[m_slice_arr, n_idx, :] - hinge_point[None, :]),
            )
            + hinge_point[None, :]
        )

    # update grid
    return grid_out.at[jnp.ix_(m_slice_arr, n_slice_arr, jnp.arange(3))].set(
        vmap(inner_func, in_axes=0, out_axes=1)(n_slice_arr)
    )


def compute_surf_c(zeta: Array) -> Array:
    r"""
    Compute the colocation points for a given grid of points on a single surface.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3]
    :return: Colocation points [m, n, 3]
    """
    return neighbour_average(zeta, axes=(0, 1))


def compute_surf_nc(zeta: Array) -> Array:
    r"""
    Compute the surface normal vectors for a given grid of points on a single surface. These have length equal to the
    area of their corresponding panel.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3].
    :return: Normal vectors [m, n, 3].
    """
    diag1 = zeta[1:, 1:, :] - zeta[:-1, :-1, :]
    diag2 = zeta[1:, :-1, :] - zeta[:-1, 1:, :]
    return jnp.cross(diag1, diag2)


def compute_c(zetas: ArrayList) -> ArrayList:
    r"""
    Compute the colocation points for a list of surface grids.
    :param zetas: Grids of points, [n_surf][zeta_m, zeta_n, 3].
    :return: Colocation points [n_surf][m, n, 3].
    """
    return ArrayList([compute_surf_c(zeta) for zeta in zetas])


def compute_nc(zetas: ArrayList) -> ArrayList:
    r"""
    Compute the surface normal vectors for a list of surface grids.
    :param zetas: Grids of points, [n_surf][zeta_m, zeta_n, 3].
    :return: Normal vectors [n_surf][m, n, 3].
    """
    return ArrayList([compute_surf_nc(zeta) for zeta in zetas])


def calculate_steady_forcing(
    zeta_b: ArrayList,
    zeta_dot_b: Optional[ArrayList],
    gamma_b: ArrayList,
    gamma_w: ArrayList,
    rho: Array,
    v_func: Callable[[Array], Array],
    v_inputs: Optional[ArrayList],
) -> ArrayList:
    r"""
    Calculate steady aerodynamic forcing for all surfaces at specified time step.
    :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3].
    :param zeta_dot_b: Bound grid velocities, [n_surf][zeta_m, zeta_n, 3].
    :param gamma_b: Bound grid circulation, [n_surf][m, n]
    :param gamma_w: Wake grid circulation, [n_surf][m, n]
    :param rho: Flow field density.
    :param v_func: Total velocity as a function of coordinate.
    :param v_inputs: Additive inputs for total velocity on bound grid vertex, used for the linear solver for custom
    perturbations.
    """

    f_steady = ArrayList([])

    if zeta_dot_b is None:
        zeta_dot_bs_: list[Optional[Array]] = [None] * len(zeta_b)
    else:
        zeta_dot_bs_ = zeta_dot_b

    if v_inputs is None:
        v_inputs_ = [None] * len(zeta_b)
    else:
        v_inputs_ = v_inputs

    for zeta_b, zeta_dot_b, gamma_b, gamma_w, v_input in zip(
        zeta_b, zeta_dot_bs_, gamma_b, gamma_w, v_inputs_
    ):
        # compute midpoints
        mp_chordwise = neighbour_average(zeta_b, axes=0)  # [gamma_m, gamma_n+1, 3]
        mp_spanwise = neighbour_average(zeta_b, axes=1)  # [gamma_m+1, gamma_n, 3]

        assert zeta_dot_b is not None

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

        # equivalent strengths of filaments
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
    gamma_b_nm1: Array,
    gamma_w_nm1: Array,
    zeta_b_n: Array,
    zeta_w_nm1: Array,
    delta_w: Optional[Array],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: bool,
    linearise_variable_wake: bool = False,
) -> tuple[Optional[Array], Array]:
    r"""
    Convect the wake at some given velocity for a single surface from timestep n-1 to timestep n. This step includes
    convection from the trailing edge and culling the downstream data.
    :param gamma_b_nm1: Bound circulation at time step n-1, [m, n].
    :param gamma_w_nm1: Wake circulation at time step n-1, [m_star, n].
    :param zeta_b_n: Bound grid at time step n, [zeta_m, zeta_n, 3].
    :param zeta_w_nm1: Wake grid at time step n-1, [zeta_m_star, zeta_n, 3].
    :param delta_w: Desired wake discretisation, [zeta_m_star, 3], or None for uniform.
    :param v_func: Function that computes the velocity as a function of coordinate, [3] -> [3].
    :param dt: Time step length.
    :param frozen_wake: If true, the grid stays constant with time. Used in the linearised case.
    :param linearise_variable_wake: If true, block gradients through the arc-length computation so that
        the re-discretisation is treated as a linear operator when differentiated.
    :return: New wake grid and circulation, [zeta_m_star, zeta_n, 3], [m_star, n].
    """

    # trailing edge positions and circulations
    zeta_te = zeta_b_n[-1, ...]  # [zeta_n, 3]
    gamma_te = gamma_b_nm1[-1, ...]  # [gamma_n]

    # variable wake discretisation also depends on the final element
    if delta_w is not None:
        zeta_base = zeta_w_nm1  # [zeta_w_m, zeta_n, 3]
        gamma_base = gamma_w_nm1  # [gamma_w_m, gamma_n]
    else:
        zeta_base = zeta_w_nm1[:-1, ...]  # [zeta_w_m - 1, zeta_n, 3]
        gamma_base = gamma_w_nm1[:-1, ...]  # [gamma_w_m - 1, gamma_n]

    # values at t=varphi+1 before re-discretisation
    gamma_w_np1 = jnp.concatenate(
        (gamma_te[None, ...], gamma_base), axis=0
    )  # [gamma_w_m+1 | gamma_w_m, gamma_n]

    # if the wake is free, this should be embedded here
    v = v_func(zeta_base)  # [zeta_w_m | zeta_w_m-1, zeta_n, 3]

    # wake coordinates at t=varphi+1 before re-discretisation
    zeta_w_np1 = jnp.concatenate(
        (zeta_te[None, :, :], zeta_base + dt * v), axis=0
    )  # [zeta_w_m+1 | zeta_w_m, zeta_n, 3]

    if delta_w is not None:
        # streamline coordinates before re-discretisation
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

        if linearise_variable_wake:
            s_zeta_w = jax.lax.stop_gradient(s_zeta_w)

        # consider gamma to be at midpoints of zeta
        s_gamma_w = neighbour_average(s_zeta_w, axes=(0, 1))

        # vertex coordinates along desired discretised streamline, [m_star + 1]
        s_zeta_w_discretisation = jnp.concatenate((jnp.zeros(1), jnp.cumsum(delta_w)))

        # midpoint coordinates along desired discretised streamline, [m_star]
        s_gamma_w_discretisation = neighbour_average(s_zeta_w_discretisation, axes=(0,))

        # re-discretise coordinates onto desired grid
        zeta_w_np1 = vmap(
            vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=1),
            in_axes=(None, None, 1),
            out_axes=2,
        )(
            s_zeta_w_discretisation, s_zeta_w.T, jnp.transpose(zeta_w_np1, (1, 2, 0))
        )  # [zeta_w_m, zeta_n, 3]

        # re-discretise gamma onto desired grid
        gamma_w_np1 = vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=1)(
            s_gamma_w_discretisation, s_gamma_w.T, gamma_w_np1.T
        )  # [zeta_w_m, zeta_n, 3]

    if frozen_wake:
        return None, gamma_w_np1
    else:
        return zeta_w_np1, gamma_w_np1


@overload
def propagate_wake(
    gamma_b_nm1: ArrayList,
    gamma_w_nm1: ArrayList,
    zeta_b_n: ArrayList,
    zeta_w_nm1: ArrayList,
    delta_w: Sequence[Optional[Array]],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: Literal[True],
    linearise_variable_wake: bool,
) -> tuple[None, ArrayList]: ...


@overload
def propagate_wake(
    gamma_b_nm1: ArrayList,
    gamma_w_nm1: ArrayList,
    zeta_b_n: ArrayList,
    zeta_w_nm1: ArrayList,
    delta_w: Sequence[Optional[Array]],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: Literal[False],
    linearise_variable_wake: bool,
) -> tuple[ArrayList, ArrayList]: ...


def propagate_wake(
    gamma_b_nm1: ArrayList,
    gamma_w_nm1: ArrayList,
    zeta_b_n: ArrayList,
    zeta_w_nm1: ArrayList,
    delta_w: Sequence[Optional[Array]],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: bool,
    linearise_variable_wake: bool = False,
) -> tuple[Optional[ArrayList], ArrayList]:
    r"""
    Convect the wake for all surfaces.
    :param gamma_b_nm1: Bound circulation at time step n-1, [n_surf][m, n].
    :param gamma_w_nm1: Wake circulation at time step n-1, [n_surf][m_star, n].
    :param zeta_b_n: Bound grid at time step n, [n_surf][zeta_m, zeta_n, 3].
    :param zeta_w_nm1: Wake grid at time step n-1, [n_surf][zeta_m_star, zeta_n, 3].
    :param delta_w: Desired wake discretisation, [n_surf][zeta_m_star, 3] or None for uniform.
    :param v_func: Function that computes the velocity, [3] -> [3].
    :param dt: Time step length.
    :param frozen_wake: If true, the grid stays constant with time, useful in the linearised case.
    :param linearise_variable_wake: If true, block gradients through the arc-length computation so that
        the re-discretisation is treated as a linear operator when differentiated with jax.jvp.
    :return: New wake grid and circulation, [n_surf][zeta_m_star, zeta_n, 3], [n_surf][m_star, n].
    """

    n_surf = len(gamma_b_nm1)
    zeta_w_np1: Optional[ArrayList] = ArrayList([]) if not frozen_wake else None
    gamma_w_np1 = ArrayList([])

    for i_surf in range(n_surf):
        surf_zeta_w, surf_gamma_w = propagate_surf_wake(
            gamma_b_nm1=gamma_b_nm1[i_surf],
            gamma_w_nm1=gamma_w_nm1[i_surf],
            zeta_b_n=zeta_b_n[i_surf],
            zeta_w_nm1=zeta_w_nm1[i_surf],
            delta_w=delta_w[i_surf],
            v_func=v_func,
            dt=dt,
            frozen_wake=frozen_wake,
            linearise_variable_wake=linearise_variable_wake,
        )
        if zeta_w_np1 is not None:
            assert surf_zeta_w is not None
            zeta_w_np1.append(surf_zeta_w)
        gamma_w_np1.append(surf_gamma_w)
    return zeta_w_np1, gamma_w_np1


def biot_savart(x: Array, y: Array) -> Array:
    r"""
    Biot-Savart kernel without any smoothing or cutoff.
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
    Differentiable function to obtain a unit vector that is defined for all ``r``. As ``r`` -> 0, the output approaches zero instead of being
    undefined.
    :param r: Vector to be normalised, [3].
    :return: Unit vector, [3].
    """
    return r / jnp.sqrt(jnp.sum(r**2) + EPSILON**2)


@make_unit_epsilon.defjvp
def smooth_unit_vector_jvp(primals, tangents):
    r"""
    Custom JVP rule for the smoothed unit vector function.
    """
    r = primals[0]
    r_dot = tangents[0]
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
    r1_x_r2_unit = r1_x_r2 / (
        jnp.inner(r1_x_r2, r1_x_r2) + EPSILON * jnp.inner(r0, r0) ** 2
    )
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
    Mirror a grid of points across a plane defined by a point and a normal vector.
    :param zeta: Grid of points, [zeta_m, zeta_n, 3].
    :param mirror_point: Point in mirror plane, [3].
    :param mirror_normal: Normal vector of mirror plane, [3]. Should be normalised.
    :return: Mirrored grid of points, [zeta_m, zeta_n, 3].
    """
    diff = zeta - mirror_point[None, None, :]  # [zeta_m, zeta_n, 3]
    diff_n = jnp.einsum("ijk,k->ij", diff, mirror_normal)  # [zeta_m, zeta_n]
    return (
        zeta - 2.0 * diff_n[:, :, None] * mirror_normal[None, None, :]
    )  # [zeta_m, zeta_n, 3]


def project_forcing_to_beam(
    f_total: ArrayList,
    rmat: Array,
    dof_mapping: ArrayList,
    x0_aero: ArrayList,
) -> Array:
    r"""
    Project aerodynamic forcing at specified time step onto the beam grid. Returned forces are in the global frame.
    :param f_total: Total force on aerodynamic grid, [n_surf][m+1, n+1, 3]
    :param rmat: Rotation matrix for each node relative to reference, [n_nodes, 3, 3].
    :param x0_aero: Reference coordinates for aerodynamic grid, [n_surf][zeta_m, zeta_n, 3].
    :param dof_mapping: Mapping between aero and beam discretisations.
    :return: Steady and unsteady forcing projected onto the beam grid, [n_nodes, 6]
    """

    n_nodes = rmat.shape[0]
    result = jnp.zeros((n_nodes, 6))

    for i_surf in range(len(f_total)):
        # rotate relative distances to get moment arms
        this_rmat = rmat[dof_mapping[i_surf], ...]  # [zeta_n, 3, 3]
        r_x0 = jnp.einsum(
            "ijk,lik->lij", this_rmat, x0_aero[i_surf]
        )  # relative distance [zeta_n, zeta_m, 3]

        result = result.at[dof_mapping[i_surf], :3].set(
            f_total[i_surf].sum(axis=0)
        )  # forcing is sum along strip [zeta_n, 3]
        result = result.at[dof_mapping[i_surf], 3:].set(
            jnp.cross(r_x0, f_total[i_surf]).sum(axis=0)
        )  # moment is cross(r, f) summed along strip [zeta_n, 3]
    return result


def cs_ang_to_cs_vel(cs_ang_t: dict[str, Array], dt: float | Array) -> dict[str, Array]:
    r"""
    Approximate control surfaces velocities from the time series of their angles using finite differences.
    :param cs_ang_t: Time history of control surface angle, {name, [n_tstep]}.
    :param dt: Time step length.
    :return: Control surface velocity, {name, [n_tstep]}.
    """
    cs_vel_t = dict()
    for k, v in cs_ang_t.items():
        n_tstep = v.shape[0]
        cs_vel_t[k] = vmap(
            lambda i_ts: finite_difference(
                i_=i_ts, data=v, delta=jnp.array(dt), axis=0
            ),
            in_axes=0,
            out_axes=0,
        )(jnp.arange(n_tstep))
    return cs_vel_t


def cs_vel_to_cs_ang(cs_vel_t: dict[str, Array], dt: float | Array) -> dict[str, Array]:
    r"""
    Approximate control surfaces angles from the time series of their velocities using finite differences.
    :param cs_vel_t: Time history of control surface velocity, {name, [n_tstep]}.
    :param dt: Time step length.
    :return: Control surface angle, {name, [n_tstep]}.
    """
    cs_ang_t = dict()
    for k, v in cs_vel_t.items():
        cs_ang_t[k] = jnp.cumsum(v, axis=0) * dt
    return cs_ang_t
