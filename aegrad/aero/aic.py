from __future__ import annotations
from typing import Sequence, Optional
from jax import Array, vmap
from jax import numpy as jnp

from aegrad.aero.kernels import KernelFunction
from aegrad.algebra.array_utils import block_axis


r"""
Functions used to create and transform AIC matrices.
"""


def _compute_aic_grid(
    c: Array,
    zeta: Array,
    kernel: KernelFunction,
):
    """
    Compute the aerodynamic influence coefficient (AIC) across grids of points.
    :param c: Collocation points, [c_m, c_n, 3].
    :param zeta: Grid points, [zeta_m, zeta_n, 3].
    :param kernel: Kernel function to compute the influence.
    :return: AIC matrix, [c_m, c_n, zeta_m, zeta_n, 3].
    """

    # create the AIC in the spanwise and chordwise directions, and combine later
    # vectors in chordwise direction [zeta_m - 1, zeta_n, 2, 3]
    m_vect = jnp.stack((zeta[:-1, :, :], zeta[1:, :, :]), axis=-2)

    # vectors in spanwise directionc [zeta_m, zeta_n - 1, 2, 3]
    n_vect = jnp.stack((zeta[:, :-1, :], zeta[:, 1:, :]), axis=-2)

    # AIC matrices have one entry per filament
    m_aic = _aic_vmap(c, m_vect, kernel)  # chordwise AIC [m, n, zeta_m - 1, zeta_n, 3]
    n_aic = _aic_vmap(c, n_vect, kernel)  # spanwise AIC [m, n, zeta_m, zeta_n - 1, 3]

    return -jnp.diff(m_aic, axis=3) + jnp.diff(n_aic, axis=2)


def _compute_aic_sys(
    cs: Sequence[Array],
    zetas: Sequence[Array],
    kernels: Sequence[KernelFunction],
    ns: Optional[Sequence[Array]] = None,
) -> list[list[Array]]:
    """
    Compute the AIC matrix for a system of elements. Returns a list of AIC matrices, one for each element.
    :param cs: List of collocation points, [][c_m, c_n, 3].
    :param zetas: List of grids, [][zeta_m, zeta_n, 3].
    :param kernels: List of kernel functions for each element.
    :param ns: Optional list of normal vectors for each element, [][c_m, c_n, 3]. If provided, the AICs will be projected
    onto these normals.
    :return: Nested sequences of AIC matrices, [][][c_m, c_n, zeta_m, zeta_n, 3], or [][][c_m, c_n, zeta_m, zeta_n] if projected onto normals.
    """
    if len(zetas) != len(kernels):
        raise ValueError(
            "Number of grids must match number of elements. "
            f"Got {len(zetas)} grids and {len(kernels)} kernels."
        )

    if ns is not None and len(ns) != len(kernels):
        raise ValueError(
            "Number of normal vectors must match number of elements. "
            f"Got {len(ns)} normals and {len(kernels)} kernels."
        )

    aic_mats = []
    for i_c, c in enumerate(cs):
        aic_mats.append([])
        for i_z, zeta, kernel in zip(range(len(zetas)), zetas, kernels):
            # compute the AIC matrix, [n_cx, n_cy, n_ex, n_ey, 3]
            aic_ = _compute_aic_grid(
                c,
                zeta,
                kernel,
            )
            if ns is not None:
                aic_ = jnp.einsum(
                    "ijklm,ijm->ijkl", aic_, ns[i_c]
                )  # project onto normals
            aic_mats[-1].append(aic_)
    return aic_mats


def _add_wake_influence(
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


def _reshape_aic_sys(aic_mat: Array) -> Array:
    r"""
    Reshape an AIC matrix such that the source and target dimensions are flattened.
    :param aic_mat: Input AIC matrix, [c_m, c_n, zeta_m, zeta_n] or [c_m, c_n, zeta_m, zeta_n, 3].
    :return: Reshaped AIC matrix, [c_m*c_n, zeta_m*zeta_n] or [c_m*c_n, zeta_m*zeta_n, 3].
    """
    shape = aic_mat.shape
    new_shape = [shape[0] * shape[1], shape[2] * shape[3]]
    if len(shape) == 5:
        new_shape.append(shape[4])
    return aic_mat.reshape(new_shape)


def _assemble_aic_sys(aic_mats: Sequence[Sequence[Array]]) -> Array:
    r"""
    Assemble a nested sequence of AIC matrices into a single AIC matrix.
    :param aic_mats: Nested sequence of AIC matrices, [][][c_m, c_n, zeta_m, zeta_n] or [][][c_m, c_n, zeta_m, zeta_n, 3].
    :return: Assembled AIC matrix. [c_tot, zeta_tot] or [c_tot, zeta_tot, 3].
    """
    aic_mats_reshaped = [
        [_reshape_aic_sys(aic) for aic in aic_row] for aic_row in aic_mats
    ]
    return block_axis(aic_mats_reshaped, axes=(0, 1))


def _compute_aic_sys_assembled(
    cs: Sequence[Array],
    zetas: Sequence[Array],
    kernels: Sequence[KernelFunction],
    ns: Optional[Sequence[Array]] = None,
) -> Array:
    """
    Compute the assembled AIC matrix for a system of elements.
    :param cs: List of collocation points, [][c_m, c_n, 3].
    :param zetas: List of grids, [][zeta_m, zeta_n, 3].
    :param kernels: List of kernel functions for each element.
    :param ns: Optional list of normal vectors for each element, [][c_m, c_n, 3]. If provided, the AICs will be projected
    onto these normals.
    :return: Full AIC matrix, [c_tot, zeta_tot, 3], or [c_tot, zeta_tot] if projected onto normals.
    """
    aic_mats = _compute_aic_sys(cs, zetas, kernels, ns)

    aic_mats_reshaped = [
        [_reshape_aic_sys(aic) for aic in aic_row] for aic_row in aic_mats
    ]

    return block_axis(aic_mats_reshaped, axes=(0, 1))


def _aic_vmap(
    c: Array,
    zeta: Array,
    kernel: KernelFunction,
) -> Array:
    """
    General AIC computation for any grid. Will vmap across first two dimensions.
    :param c: Collocation points, [c_m, c_n, 3].
    :param zeta: Grid points, [zeta_m, zeta_n, 3].
    :param kernel: Kernel function to compute the influence.
    :return: AIC matrix, [c_m, c_n, zeta_m, zeta_n, 3].
    """

    return vmap(
        vmap(vmap(vmap(kernel, (0, None), 0), (1, None), 1), (None, 0), 2), (None, 1), 3
    )(c, zeta)
