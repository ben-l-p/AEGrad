from typing import Literal, Callable
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jacrev, Array

from aegrad.aero.data_structures import DynamicAeroCase
from aero.utils import KernelFunction, _get_surf_c, _get_surf_nc, mirror_grid
from algebra.array_utils import ArrayList


class DynamicAeroCaseGrad(DynamicAeroCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.zeta_b_rings: ArrayList = ArrayList(
            [vmap(self._grid_to_ring_decomp, 0, 0)(zeta_b) for zeta_b in self.zeta_b]
        )  # [n_surf][n_ts, m, n, 2, 2, 3]

        self.zeta_w_rings: ArrayList = ArrayList(
            [vmap(self._grid_to_ring_decomp, 0, 0)(zeta_w) for zeta_w in self.zeta_w]
        )  # [n_surf][n_ts, m_star, n, 2, 2, 3]

        self.gamma_b_slices: list[slice] = self._make_slices(self.gamma_b.index_all(0))
        self.zeta_dof_target_slices: list[slice] = self._make_slices(
            self.target_zeta.index_all(0)
        )
        self.zeta_dof_source_slices: list[slice] = self._make_slices(
            self.source_zeta.index_all(0)
        )

    @property
    def source_rings(self) -> ArrayList:
        return ArrayList([*self.zeta_b_rings, *self.zeta_w_rings])

    @property
    def target_rings(self) -> ArrayList:
        return self.zeta_b_rings

    @property
    def source_zeta(self) -> ArrayList:
        return ArrayList([*self.zeta_b, *self.zeta_w])

    @property
    def target_zeta(self) -> ArrayList:
        return self.zeta_b

    def is_wake(self, i_surf: int) -> bool:
        return i_surf >= self.n_surf

    @staticmethod
    def _make_slices(arrs: ArrayList) -> list[slice]:
        slices = []
        cnt = 0
        for arr in arrs:
            sz = arr.size
            slices.append(slice(cnt, cnt + sz))
            cnt += sz
        return slices

    @staticmethod
    def _grid_to_ring_decomp(
        zeta: Array,
    ) -> Array:
        r"""
        Decompose a grid into rings for AIC computation. This is needed for the Jacobian of the AIC with respect to the grid coordinates.
        :param zeta: [zeta_m, zeta_n, 3] grid coordinates
        :return: [m, n, 2, 2, 3] ring decomposition of the grid
        """
        return jnp.stack(
            (
                jnp.stack((zeta[:-1, :-1, :], zeta[1:, :-1, :]), axis=2),
                jnp.stack((zeta[:-1, 1:, :], zeta[1:, 1:, :]), axis=2),
            ),
            axis=3,
        )  # [m, n, 2, 2, 3]

    def _aic_entry(
        self,
        target_ring: Array,
        source_ring: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> Array:
        r"""
        Compute an AIC entry for a ring pair.
        :param source_ring: [2, 2, 3] coordinates of the source ring vertices
        :param target_ring: [2, 2, 3] coordinates of the target ring vertices
        :param kernel: kernel function
        :param project_to_normal: if true, the influence coefficient is projected onto the normal vector at the target ring
        :return: Influence coefficient for the target ring from the source ring projected onto the normal, [] or [3]
        depending on project_to_normal
        """
        c = _get_surf_c(target_ring)  # [3]
        aic = self._compute_aic_grid(c, source_ring, kernel)[0, 0, 0, 0, :]  # [3]
        if self.mirror_point is not None and self.mirror_normal is not None:
            source_ring_mirrored = mirror_grid(
                zeta=source_ring,
                mirror_point=self.mirror_point,
                mirror_normal=self.mirror_normal,
            )
            aic += self._compute_aic_grid(c, source_ring_mirrored, kernel)[
                0, 0, 0, 0, :
            ]  # [3]

        if project_to_normal:
            nc = _get_surf_nc(target_ring)[0, 0, :]  # [3]
            return jnp.inner(aic, nc)  # []
        else:
            return aic  # [3]

    def _d_aic_d_target(
        self,
        target_ring: Array,
        source_ring: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> Array:
        r"""
        Compute the Jacobian of the AIC entry with respect to the target ring coordinates.
        :param source_ring: [2, 2, 3]
        :param target_ring: [2, 2, 3]
        :param kernel: kernel function
        :return: [2, 2, 3] or [3, 2, 2, 3], derivatives with respect to target ring coordinates. Includes effect of
        perturbations in the normal vector if project_to_normal is true.
        """
        return jacrev(self._aic_entry, argnums=0)(
            target_ring,
            source_ring,
            kernel=kernel,
            project_to_normal=project_to_normal,
        )

    def _d_aic_d_source(
        self,
        target_ring: Array,
        source_ring: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> Array:
        r"""
        Compute the Jacobian of the AIC entry with respect to the source ring coordinates.
        :param source_ring: [2, 2, 3]
        :param target_ring: [2, 2, 3]
        :param kernel: kernel function
        :return: [2, 2, 3] or [3, 2, 2, 3], derivatives with respect to target ring coordinates. Includes effect of
        perturbations in the normal vector if project_to_normal is true.
        """
        return jacrev(self._aic_entry, argnums=1)(
            target_ring,
            source_ring,
            kernel=kernel,
            project_to_normal=project_to_normal,
        )

    @staticmethod
    def _ring_decomp_to_grid(
        aic_ring: Array,
        wrt: Literal["target", "source"],
    ) -> Array:
        r"""
        Recombine a ring-wise matrix back onto a full grid
        :param aic_ring: [m_t, n_t, m_s, n_s, 2, 2, 3] matrix defined on rings
        :return: [m_t, n_t, (m_t + 1), (n_t + 1), 3] panel scalars gradient with respect to target grid coordinates, or
        [m_t, n_t, (m_s + 1), (n_s + 1), 3] with respect to source grid coordinates.
        """
        m_t, n_t, m_s, n_s = aic_ring.shape[:4]
        match wrt:
            case "target":
                temp = aic_ring.sum(axis=(2, 3))  # [m_t, n_t, 2, 2, 3]
                aic_grid = jnp.zeros((m_t, n_t, m_t + 1, n_t + 1, 3))
                aic_grid = aic_grid.at[:, :, :-1, :-1, :].add(temp[:, :, 0, 0, :])
                aic_grid = aic_grid.at[:, :, 1:, :-1, :].add(temp[:, :, 1, 0, :])
                aic_grid = aic_grid.at[:, :, :-1, 1:, :].add(temp[:, :, 0, 1, :])
                aic_grid = aic_grid.at[:, :, 1:, 1:, :].add(temp[:, :, 1, 1, :])

            case "source":
                aic_grid = jnp.zeros((m_t, n_t, m_s + 1, n_s + 1, 3))
                aic_grid = aic_grid.at[:, :, :-1, :-1, :].add(
                    aic_ring[:, :, :, :, 0, 0, :]
                )
                aic_grid = aic_grid.at[:, :, 1:, :-1, :].add(
                    aic_ring[:, :, :, :, 1, 0, :]
                )
                aic_grid = aic_grid.at[:, :, :-1, 1:, :].add(
                    aic_ring[:, :, :, :, 0, 1, :]
                )
                aic_grid = aic_grid.at[:, :, 1:, 1:, :].add(
                    aic_ring[:, :, :, :, 1, 1, :]
                )
            case _:
                raise ValueError(f"Invalid value for wrt: {wrt}")
        return aic_grid

    @classmethod
    def _d_aic_gamma(
        cls,
        gamma_b: Array,
        zeta_target_rings: Array,
        zeta_source_rings: Array,
        grad_func: Callable[[Array, Array], Array],
        wrt: Literal["target", "source"],
    ) -> Array:
        r"""
        Compute :math:`\frac{d \mathbf{A}}{d \zeta_b} \cdot \Gamma` for a surface pair bound grid. This is needed for the
        :param gamma_b: Bound circulation, [m, n]
        :param zeta_target_rings: Target rings, [m_t n_t, 2, 2, 3]
        :param zeta_source_rings: Source rings, [m_s, n_s, 2, 2, 3]
        :return: Derivative of AIC circulation product with respect to target and source bound grid coordinates,
        [m_t, n_t, zeta_target_m, zeta_target_n, 3] or [m_t, n_t, 3, zeta_target_m, zeta_target_n, 3], depending on normal
        projection.
        """

        d_aic = vmap(
            vmap(
                vmap(
                    vmap(
                        grad_func,
                        (0, None),
                        0,
                    ),
                    (1, None),
                    1,
                ),
                (None, 0),
                2,
            ),
            (None, 1),
            3,
        )(zeta_target_rings, zeta_source_rings)  # [m_t, n_t, m_s, n_s, 2, 2, 3]

        d_aic_gamma = jnp.einsum(
            "ijklmno,kl->ijklmno", d_aic, gamma_b
        )  # [m_t, n_t, m_s, n_s, 2, 2, 3]
        return cls._ring_decomp_to_grid(d_aic_gamma, wrt=wrt)

    def _d_aic_gamma_d_grids(
        self,
        gamma: Array,
        zeta_target_rings: Array,
        zeta_source_rings: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> tuple[Array, Array]:
        r"""
        Compute the contribution to the Jacobian of the AIC circulation product with respect to target and source bound grid
        coordinates.
        :param gamma: Circulation on the source grid, [m_s, n_s]
        :param zeta_target_rings: Target rings for the AIC computation, [m_t, n_t, 2, 2, 3]
        :param zeta_source_rings: Source rings for the AIC computation, [m_s, n_s, 2, 2, 3]
        :param kernel: Kernel function
        :return: AIC circulation product Jacobian contributions with respect to target and source bound grid coordinates,
        [m_t, n_t, zeta_target_m, zeta_target_n, 3] and [m_t, n_t, zeta_source_m, zeta_source_n, 3] if project to normals,
        otherwise [m_t, n_t, 3, zeta_target_m, zeta_target_n, 3] and [m_t, n_t, 3, zeta_source_m, zeta_source_n, 3]
        """

        d_aic_gamma_d_zeta_target = self._d_aic_gamma(
            zeta_target_rings=zeta_target_rings,
            zeta_source_rings=zeta_source_rings,
            gamma_b=gamma,
            grad_func=partial(
                self._d_aic_d_target,
                kernel=kernel,
                project_to_normal=project_to_normal,
            ),
            wrt="target",
        )
        d_aic_gamma_d_zeta_source = self._d_aic_gamma(
            zeta_target_rings=zeta_target_rings,
            zeta_source_rings=zeta_source_rings,
            gamma_b=gamma,
            grad_func=partial(
                self._d_aic_d_source,
                kernel=kernel,
                project_to_normal=project_to_normal,
            ),
            wrt="source",
        )
        return d_aic_gamma_d_zeta_target, d_aic_gamma_d_zeta_source

    def _d_v_bc_d_zeta(
        self,
        zeta_target_rings: Array,
    ) -> Array:
        r"""
        Compute the contribution to the Jacobian of the boundary condition velocity projected onto the normals, with respect
        to the target grid coordinates.
        :param zeta_target_rings: Target rings for the AIC computation, [m_t, n_t, 2, 2, 3]
        :return: Boundary condition velocity Jacobian contribution with respect to target bound grid coordinates,
        [m_t, n_t, 2, 2, 3]
        """

        def v_bc_entry(target_ring: Array) -> Array:
            c = _get_surf_c(target_ring)[0, 0, :]  # [3]
            nc = _get_surf_nc(target_ring)[0, 0, :]  # [3]
            v = self.flowfield(c, self.t)  # [3]
            return jnp.inner(v, nc)  # []

        def d_v_bc_entry_d_target(target_ring: Array) -> Array:
            return jacrev(v_bc_entry)(target_ring)  # [2, 2, 3]

        # we don't assemble this as it would be sparse
        return vmap(vmap(d_v_bc_entry_d_target, 0, 0), 1, 1)(
            zeta_target_rings
        )  # [m_t, n_t, 2, 2, 3]

    def _d_gamma_b_d_zeta_b(
        self,
        static: bool,
        i_ts: int,
    ) -> Array:

        gamma = [*self.gamma_b, *self.gamma_w]

        if static:
            n_zeta_dofs = self.n_bound_zeta_dofs
        else:
            raise NotImplementedError("Dynamic case not implemented yet")

        d_gamma_b_zeta_b = jnp.zeros((self.n_bound_panels, n_zeta_dofs))  # full matrix

        for i_target in range(self.n_surf):
            for i_source in range(2 * self.n_surf):
                d_aic_gamma_d_zeta_target, d_aic_gamma_d_zeta_source = (
                    self._d_aic_gamma_d_grids(
                        gamma=gamma[i_source],
                        zeta_target_rings=self.target_rings[i_target],
                        zeta_source_rings=self.source_rings[i_source],
                        kernel=self.kernels[i_source],
                        project_to_normal=True,
                    )
                )

                # shrink to wake if static
                if self.is_wake(i_source) and static:
                    d_aic_gamma_d_zeta_source = d_aic_gamma_d_zeta_source.sum(
                        axis=2, keepdims=True
                    )

                # perturbations on the source rings
                d_gamma_b_zeta_b = d_gamma_b_zeta_b.at[
                    self.gamma_b_slices[i_target], self.zeta_dof_source_slices[i_source]
                ].add(
                    d_aic_gamma_d_zeta_source.reshape(self.gamma_b[i_target].size, -1)
                )

                # perturbations on the target rings
                d_gamma_b_zeta_b = d_gamma_b_zeta_b.at[
                    self.gamma_b_slices[i_target], self.zeta_dof_target_slices[i_target]
                ].add(
                    d_aic_gamma_d_zeta_target.reshape(
                        self.gamma_b[i_target].size, self.zeta_b[i_target].size
                    )
                )

            # add pertubations in boundary condition (has no source)
            d_v_bc_d_zeta = self._d_v_bc_d_zeta(
                zeta_target_rings=self.zeta_b_rings[i_target],
            )  # [m_t, n_t, 2, 2, 3]

            m_t, n_t = self.gamma_b[i_target].shape
            assemble_d_v_bc = jnp.zeros((m_t, n_t, m_t + 1, n_t + 1, 3))

            assemble_d_v_bc = assemble_d_v_bc.at[:, :, :-1, :-1, :].add(
                d_v_bc_d_zeta[:, :, 0, 0, :]
            )
            assemble_d_v_bc = assemble_d_v_bc.at[:, :, 1:, :-1, :].add(
                d_v_bc_d_zeta[:, :, 1, 0, :]
            )
            assemble_d_v_bc = assemble_d_v_bc.at[:, :, :-1, 1:, :].add(
                d_v_bc_d_zeta[:, :, 0, 1, :]
            )
            assemble_d_v_bc = assemble_d_v_bc.at[:, :, 1:, 1:, :].add(
                d_v_bc_d_zeta[:, :, 1, 1, :]
            )

            d_gamma_b_zeta_b = d_gamma_b_zeta_b.at[
                self.gamma_b_slices[i_target], self.zeta_dof_target_slices[i_target]
            ].add(assemble_d_v_bc.reshape(self.gamma_b[i_target].size, -1))

        return -jsp.linalg.lu_solve(
            (self.aic_lu[i_ts, ...], self.aic_piv[i_ts, ...]), d_gamma_b_zeta_b
        )


# def _d_v_mp_zeta_b(
#     zeta_b_to_x: Literal["chordwise_mp", "spanwise_mp", "direct"],
#     gamma_b: ArrayList,
#     gamma_w: ArrayList,
#     zeta_b: ArrayList,
#     zeta_w: ArrayList,
#     kernels: Sequence[KernelFunction],
#     flowfield: FlowField,
#     mirror_point: Optional[Array],
#     mirror_normal: Optional[Array],
#     t: Array,
# ) -> Array:
#     r"""
#     :param gamma_b:
#     :param gamma_w:
#     :param zeta_b:
#     :param zeta_w:
#     :param kernels:
#     :param flowfield:
#     :param mirror_point:
#     :param mirror_normal:
#     :param t:
#     :return:
#     """
#
#     def func_zeta_b_to_x(zeta_b_: Array) -> Array:
#         match zeta_b_to_x:
#             case "chordwise_mp":
#                 return 0.5 * (zeta_b_[:-1, :, :] + zeta_b_[1, :, :])
#             case "spanwise_mp":
#                 return 0.5 * (zeta_b_[:-1, :, :] + zeta_b_[1, :, :])
#             case "direct":
#                 return zeta_b_
#             case _:
#                 raise ValueError(f"Invalid value for zeta_b_to_x: {zeta_b_to_x}")
#
#     x_target = (
#         ArrayList([func_zeta_b_to_x(zeta_b_) for zeta_b_ in zeta_b])
#         .flatten()
#         .reshape(-1, 3)
#     )  # [n_x, 3]
#
#     def get_v_inf_grad(x_target: Array) -> Array:
#         r"""
#         :param x_target: Coordinate, [3]
#         :return: Gradient of the freestream velocity with respect to the coordinate, [3, 3]
#         """
#         return jacrev(flowfield, argnums=0)(x_target, t)
#
#     d_v_inf_d_x = vmap(get_v_inf_grad, 0, 0)(x_target)  # [n_x, 3, 3]
#
#
# def _d_f_steady_d_zeta_b():
#     pass
