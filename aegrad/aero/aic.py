from typing import Optional, Sequence

from jax import Array, vmap
from jax import numpy as jnp
import jax

from aero.utils import KernelFunction, mirror_grid
from algebra.array_utils import ArrayList, block_axis

# aim - rewrite solve routine in a way where performance may be degraded, but AD is efficient

BATCH_SIZE = 128


def compute_aic_grid(
    c: Array,
    n: Optional[Array],
    zeta: Array,
    kernel: KernelFunction,
    batch_size: Optional[int] = BATCH_SIZE,
):
    """
    Compute the aerodynamic influence coefficient (AIC) across grids of points.

    When normal is provided, fuses the dot product inside each map step so the
    trailing 3-component axis is never accumulated, saving 3x memory.
    :param c: Collocation points, [c_m, c_n, 3].
    :param n: Normal vectors at collocation points, [c_m, c_n, 3], or None.
    :param zeta: Grid vertices, [m+1, n+1, 3].
    :param kernel: Kernel function to compute the influence.
    :param batch_size: Passed to lax.map.
    :return: [c_m, c_n, m, n, 3] if normal is None, else [c_m, c_n, m, n].
    """
    c_m, c_n = c.shape[:2]
    m_panels, n_panels = zeta.shape[0] - 1, zeta.shape[1] - 1

    m_vect_flat = jnp.stack((zeta[:-1, :, :], zeta[1:, :, :]), axis=-2).reshape(
        -1, 2, 3
    )
    n_vect_flat = jnp.stack((zeta[:, :-1, :], zeta[:, 1:, :]), axis=-2).reshape(
        -1, 2, 3
    )

    @jax.checkpoint
    def row(args: tuple) -> Array:
        ci, ni = args
        m_infl = vmap(kernel, (None, 0), 0)(ci, m_vect_flat)  # [m*(n+1), 3]
        m_infl_ni = jnp.dot(m_infl, ni).reshape(m_panels, n_panels + 1)  # [m, n+1]
        n_infl = vmap(kernel, (None, 0), 0)(ci, n_vect_flat)  # [(m+1)*n, 3]
        n_infl_ni = jnp.dot(n_infl, ni).reshape(m_panels + 1, n_panels)  # [m+1, n]
        return -jnp.diff(m_infl_ni, axis=1) + jnp.diff(n_infl_ni, axis=0)  # [m, n]

    return jax.lax.map(
        row, (c.reshape(-1, 3), n.reshape(-1, 3)), batch_size=batch_size
    ).reshape(c_m, c_n, m_panels, n_panels)


def compute_aic_sys(
    zetas: ArrayList,
    cs: ArrayList,
    ns: ArrayList,
    kernels: Sequence[KernelFunction],
    mirror_point: Optional[Array] = None,
    mirror_normal: Optional[Array] = None,
) -> list[list[Array]]:
    """
    Compute the AIC matrix for a system of elements. Returns a list of AIC matrices, one for each element.
    :param zetas: List of source points to compute the AIC from, [n_surf][zeta_m, zeta_n, 3].
    :param cs: List of target points to compute the AIC at, [n_surf][c_m, c_n, 3].
    :param ns: Bound n vectors, [m, n, 3]. If None, no projection will be done.
    :param kernels: List of kernel functions to use for each source surface, [n_surf].
    :param mirror_normal: Normal vector to mirror across for image method, [3]. If None, no mirroring will be done.
    :param mirror_point: Mirror point for image method, [n_surf].
    :return: Nested sequences of AIC matrices, [n_target][n_source][c_m, c_n, c_m, c_n, 3], or [n_target][n_source][c_m, c_n, c_m, c_n] if projected onto normals.
    """

    aic_mats = []
    for c, n in zip(cs, ns):
        aic_mats.append([])
        for zeta, kernel in zip(zetas, kernels):
            # compute the AIC matrix, [n_cx, n_cy, n_ex, n_ey, 3]
            aic_ = compute_aic_grid(
                c=c,
                n=n,
                zeta=zeta,
                kernel=kernel,
            )

            if mirror_point is not None and mirror_normal is not None:
                # add influence from mirrored grid, if specified
                zeta_mirror = mirror_grid(
                    zeta=zeta,
                    mirror_point=mirror_point,
                    mirror_normal=mirror_normal,
                )
                aic_ -= compute_aic_grid(c=c, n=n, zeta=zeta_mirror, kernel=kernel)
            aic_mats[-1].append(aic_)
    return aic_mats


def reshape_aic_sys(aic_mat: Array) -> Array:
    r"""
    Reshape an AIC matrix such that the source and target dimensions are flattened.
    :param aic_mat: Input AIC matrix, [c_m, c_n, zeta_m, zeta_n] or [c_m, c_n, zeta_m, zeta_n, 3].
    :return: Reshaped AIC matrix, [c_m*c_n, zeta_m*zeta_n] or [c_m*c_n, zeta_m*zeta_n, 3].
    """
    shape = aic_mat.shape
    return aic_mat.reshape([shape[0] * shape[1], shape[2] * shape[3]])


def assemble_aic_sys(aic_mats: Sequence[Sequence[Array]]) -> Array:
    r"""
    Assemble a nested sequence of AIC matrices into a single AIC matrix.
    :param aic_mats: Nested sequence of AIC matrices, [][][c_m, c_n, zeta_m, zeta_n] or [][][c_m, c_n, zeta_m, zeta_n, 3].
    :return: Assembled AIC matrix. [c_tot, zeta_tot] or [c_tot, zeta_tot, 3].
    """
    aic_mats_reshaped = [
        [reshape_aic_sys(aic) for aic in aic_row] for aic_row in aic_mats
    ]
    return block_axis(aic_mats_reshaped, axes=(0, 1))


def compute_aic_solve(
    cs: ArrayList,
    ns: ArrayList,
    zetas_b: ArrayList,
    zetas_w: Optional[ArrayList],
    kernels_b: Sequence[KernelFunction],
    kernels_w: Optional[Sequence[KernelFunction]],
    mirror_point: Optional[Array],
    mirror_normal: Optional[Array],
) -> Array:

    aic_b_mats = compute_aic_sys(
        cs=cs,
        ns=ns,
        zetas=zetas_b,
        kernels=kernels_b,
        mirror_point=mirror_point,
        mirror_normal=mirror_normal,
    )

    if zetas_w is not None:
        aic_w_mats = compute_aic_sys(
            cs=cs,
            ns=ns,
            zetas=zetas_w,
            kernels=kernels_w,
            mirror_point=mirror_point,
            mirror_normal=mirror_normal,
        )

        aic_b_mats = add_wake_influence(aic_b_mats, aic_w_mats)

    return assemble_aic_sys(aic_b_mats)


def add_wake_influence(
    aic_bs: list[list[Array]], aic_ws: list[list[Array]]
) -> list[list[Array]]:
    r"""
    Lump the wake influence onto the last column of the bound AIC matrices.
    :param aic_bs: Bound influence matrices, [][][c_m, c_n, zeta_m, zeta_n, 3].
    :param aic_ws: Wake influence matrices, [][][c_m, c_n, zeta_m_star, zeta_n, 3].
    :return: Updated bound influence matrices, [][][c_m, c_n, zeta_m, zeta_n, 3].
    """
    for i in range(len(aic_bs)):
        for j in range(len(aic_bs[i])):
            aic_bs[i][j] = (
                aic_bs[i][j].at[:, :, -1, :].add(jnp.sum(aic_ws[i][j], axis=2))
            )
    return aic_bs


def v_ind_vmap(
    c: Array,
    zeta: Array,
    gamma: Array,
    kernel: KernelFunction,
    batch_size: Optional[int] = BATCH_SIZE,
) -> Array:
    """
    Compute einsum("ijklm,kl->ijm", aic_vmap(c, zeta, kernel), gamma) without
    materializing the full AIC. Contracts with gamma inside each lax.map step so
    the per-row intermediate shrinks from [zeta_m*zeta_n, 3] to [3].
    :param c: Collocation points, [c_m, c_n, 3].
    :param zeta: Filament grid, [zeta_m, zeta_n, 2, 3].
    :param gamma: Weights to contract with, [zeta_m, zeta_n].
    :param kernel: Kernel function.
    :param batch_size: Passed to lax.map.
    :return: [c_m, c_n, 3].
    """
    c_m, c_n = c.shape[:2]
    c_flat = c.reshape(-1, 3)
    zeta_flat = zeta.reshape(-1, 2, 3)
    gamma_flat = gamma.ravel()  # [zeta_m * zeta_n]

    @jax.checkpoint
    def row(ci: Array) -> Array:
        influence = vmap(kernel, (None, 0), 0)(ci, zeta_flat)  # [zeta_m * zeta_n, 3]
        return jnp.einsum("lm,l->m", influence, gamma_flat)  # [3]

    result = jax.lax.map(row, c_flat, batch_size=batch_size)  # [c_m * c_n, 3]
    return result.reshape(c_m, c_n, 3)


def compute_v_ind[T](
    cs: T,
    zetas: ArrayList,
    gammas: ArrayList,
    kernels: Sequence[KernelFunction],
    batch_size: Optional[int] = BATCH_SIZE,
) -> T:
    """
    Compute einsum("ijklm,kl->ijm", compute_aic_grid(c, None, zeta, kernel), gamma)
    without materializing the full [c_m, c_n, m, n, 3] AIC.

    The diff structure of compute_aic_grid is absorbed into gamma via the adjoint-diff
    identity: diff(AIC, axis) @ gamma == AIC @ adj_diff(gamma), so each filament
    matvec is fused at O(zeta_m * zeta_n) peak memory per step.
    :param cs: Collocation points, [c_m, c_n, 3].
    :param zetas: Grid vertices, [m+1, n+1, 3].
    :param gammas: Circulation strengths, [m, n].
    :param kernels: Kernel function.
    :param batch_size: Passed to lax.map.
    :return: [c_m, c_n, 3].
    """

    cs_ = ArrayList([cs]) if isinstance(cs, Array) else cs

    v = ArrayList([])
    for c in cs_:
        v.append(jnp.zeros_like(c))
        for zeta, gamma, kernel in zip(zetas, gammas, kernels):
            m_vect = jnp.stack(
                (zeta[:-1, :, :], zeta[1:, :, :]), axis=-2
            )  # [m, n+1, 2, 3]
            n_vect = jnp.stack(
                (zeta[:, :-1, :], zeta[:, 1:, :]), axis=-2
            )  # [m+1, n, 2, 3]

            gamma_eff_m = jnp.diff(jnp.pad(gamma, ((0, 0), (1, 1))), axis=1)  # [m, n+1]
            gamma_eff_n = -jnp.diff(
                jnp.pad(gamma, ((1, 1), (0, 0))), axis=0
            )  # [m+1, n]

            v[-1] += v_ind_vmap(
                c, m_vect, gamma_eff_m, kernel, batch_size
            ) + v_ind_vmap(c, n_vect, gamma_eff_n, kernel, batch_size)

    return v[0] if isinstance(cs, Array) else v
