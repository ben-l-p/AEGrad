from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING, OrderedDict

from jax import Array, numpy as jnp

if TYPE_CHECKING:
    from aegrad.aero.data_structures import DynamicAeroCase
from aegrad.algebra.array_utils import ArrayList, ArrayListShape, vect_to_arrs
from aegrad.plotting.aerogrid import plot_grid_to_vtk
from aegrad.utils.utils import make_pytree
from aegrad.utils.data_structures import DesignVariables
from aegrad.utils.print_utils import warn


@dataclass(frozen=True)
class AeroGradsToCompute:
    x0_aero: bool = True
    flowfield: bool = False
    cs_ang_t: bool = False
    cs_vel_t: bool = False


@make_pytree
class AeroStates:
    def __init__(
        self,
        gamma_b: ArrayList,
        gamma_w: ArrayList,
        gamma_b_dot: ArrayList,
        zeta_w: ArrayList,
    ) -> None:
        self.gamma_b: ArrayList = gamma_b
        self.gamma_w: ArrayList = gamma_w
        self.gamma_b_dot: ArrayList = gamma_b_dot
        self.zeta_w: ArrayList = zeta_w

    def shapes(self) -> OrderedDict[str, Optional[tuple[int, ...] | ArrayListShape]]:
        r"""
        Obtain the shapes of all arrays within the data structure.
        :return: Dictionary of name - shape pairs of all arrays within the data structure.
        """
        return OrderedDict(
            gamma_b=self.gamma_b.shape,
            gamma_w=self.gamma_w.shape,
            gamma_b_dot=self.gamma_b_dot.shape,
            zeta_w=self.zeta_w.shape,
        )

    @staticmethod
    def from_vector(
        vect: Array,
        shapes: OrderedDict[str, Optional[tuple[int, ...] | ArrayListShape]],
    ) -> AeroStates:
        return AeroStates(**vect_to_arrs(vect, shapes))

    def ravel(self) -> Array:
        r"""
        Ravel the data structure to a vector in a given order.
        :return: Data vector
        """

        return jnp.concatenate(
            [
                self.gamma_b.ravel(),
                self.gamma_w.ravel(),
                self.gamma_b_dot.ravel(),
                self.zeta_w.ravel(),
            ]
        )

    @property
    def n_states(self) -> int:
        return (
            self.gamma_b.size
            + self.gamma_w.size
            + self.gamma_b_dot.size
            + self.zeta_w.size
        )

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ()

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "gamma_b", "gamma_w", "gamma_b_dot", "zeta_w"


@make_pytree
class AeroDesignVariables(DesignVariables):
    def __init__(
        self,
        x0_aero: Optional[ArrayList],
        flowfield: Optional[dict[str, Array]],
        cs_ang_t: Optional[dict[str, Array]],
        cs_vel_t: Optional[dict[str, Array]],
        f_shape: tuple[int, ...],
    ):
        super().__init__()
        self.x0_aero: Optional[ArrayList] = x0_aero
        self.flowfield: Optional[dict[str, Array]] = flowfield
        self.cs_ang_t: Optional[dict[str, Array]] = cs_ang_t
        self.cs_vel_t: Optional[dict[str, Array]] = cs_vel_t

        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = prod(f_shape)

        self.shapes: dict[
            str,
            Optional[tuple[int, ...] | ArrayListShape]
            | dict[str, tuple[int, ...] | ArrayListShape],
        ] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def __iadd__(self, other: AeroDesignVariables) -> AeroDesignVariables:
        if self.x0_aero is not None:
            assert other.x0_aero is not None
            self.x0_aero = ArrayList(
                [self.x0_aero[i] + other.x0_aero[i] for i in range(len(self.x0_aero))]
            )
        if self.flowfield is not None:
            assert other.flowfield is not None
            for k in self.flowfield.keys():
                self.flowfield[k] += other.flowfield[k]
        if self.cs_ang_t is not None:
            assert other.cs_ang_t is not None
            for k in self.cs_ang_t.keys():
                self.cs_ang_t[k] += other.cs_ang_t[k]
        if self.cs_vel_t is not None:
            assert other.cs_vel_t is not None
            for k in self.cs_vel_t.keys():
                self.cs_vel_t[k] += other.cs_vel_t[k]
        return self

    def get_cs_n(
        self,
        i_ts: int,
        dv_full: AeroDesignVariables,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        r"""
        Obtain the angles and velocities for all control surfaces at the current timestep.
        :param i_ts: Timestep index.
        :param dv_full: Full design variables. This is used to substitute a non-differentiable value when the control
        inputs are chosen to be omitted from the design variables.
        :return: Dictionary of {name: value} pairs for control angles and velocities.
        """

        # get control surface deflections from design variables
        assert dv_full.cs_ang_t is not None and dv_full.cs_vel_t is not None
        cs_ang_n = {
            k: v[i_ts, ...]
            for k, v in (
                self.cs_ang_t if self.cs_ang_t is not None else dv_full.cs_ang_t
            ).items()
        }

        cs_vel_n = {
            k: v[i_ts, ...]
            for k, v in (
                self.cs_vel_t if self.cs_vel_t is not None else dv_full.cs_vel_t
            ).items()
        }

        return cs_ang_n, cs_vel_n

    def premultiply_adj(self, adj: Array) -> AeroDesignVariables:
        return AeroDesignVariables(
            x0_aero=ArrayList(
                [
                    jnp.einsum("ij,j...->i...", adj, self.x0_aero[i])
                    for i in range(len(self.x0_aero))
                ]
            )
            if self.x0_aero is not None
            else None,
            flowfield={
                k: jnp.einsum("ij,j...->i...", adj, v)
                for k, v in self.flowfield.items()
            }
            if self.flowfield is not None
            else None,
            cs_ang_t={
                k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.cs_ang_t.items()
            }
            if self.cs_ang_t is not None
            else None,
            cs_vel_t={
                k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.cs_vel_t.items()
            }
            if self.cs_vel_t is not None
            else None,
            f_shape=(adj.shape[1],),
        )

    def get_vars(self) -> dict[str, Optional[Array | ArrayList | dict[str, Array]]]:
        return {
            "x0_aero": self.x0_aero,
            "flowfield": self.flowfield,
            "cs_ang_t": self.cs_ang_t,
            "cs_vel_t": self.cs_vel_t,
        }

    def plot(
        self,
        case: DynamicAeroCase,
        rmat_nodal: Optional[ArrayList],
        directory: os.PathLike | str,
    ) -> Sequence[Path]:
        if self.f_size != 1:
            raise ValueError("Can only plot gradients for scalar objective functions.")

        if case.n_tstep != 1:
            raise ValueError("Can only plot gradients for singe timestep cases.")

        if self.x0_aero is None:
            warn(
                "Aerodynamic grid gradient not computed. Skipping grid gradient plotting."
            )
            return []

        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        paths = []

        for i_surf in range(case.n_surf):
            bound_filename = Path(directory).joinpath(
                case.surf_b_names[i_surf] + "_gradient"
            )

            if rmat_nodal is not None:
                d_x0_aero: Array = jnp.einsum(
                    "ijk,...lik->lij", rmat_nodal[i_surf], self.x0_aero[i_surf]
                )
            else:
                d_x0_aero = self.x0_aero[i_surf]

            paths.append(
                plot_grid_to_vtk(
                    case.zeta_b[i_surf],
                    bound_filename,
                    None,
                    node_vector_data={
                        "x0_aero": d_x0_aero,
                    },
                    cell_scalar_data={},
                )
            )

        return paths

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "x0_aero", "flowfield", "cs_ang_t", "cs_vel_t", "mapping"

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "f_size", "f_shape", "n_x", "shapes"
