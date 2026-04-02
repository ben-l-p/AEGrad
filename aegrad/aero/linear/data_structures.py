from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Sequence

from jax import Array, numpy as jnp

from aero.data_structures import AeroSnapshot
from algebra.array_utils import ArrayList, ArrayListShape
from plotting.pvd import write_pvd


@dataclass
class _LinearComponent:
    r"""
    Data class to hold information about components of a linear system.
    :param enabled: Whether this component is enabled
    :param slices: Optional sequence of slices to extract this component from a flattened array
    :param shapes: Optional sequence of arr_list_shapes for unflattening this component, [n_surf][m, varphi, ...]
    """

    enabled: bool
    slices: Optional[Sequence[slice]]
    shapes: Optional[Sequence[tuple[int, ...]] | ArrayListShape]


@dataclass
class _SliceEntry:
    r"""
    Data class to hold information about a single slice entry
    :param name: Name of the component
    :param enabled: Whether this component is enabled
    :param shapes: Optional sequence of arr_list_shapes for unflattening this component, [n_surf][m, varphi, ...]
    """

    name: str
    enabled: bool
    shapes: Optional[Sequence[tuple[int, ...]] | ArrayListShape]


@dataclass
class InputSlices:
    r"""
    Data class to hold linear components for input components
    :param zeta_b: Slices for bound grid coordinates
    :param zeta_b_dot: Slices for bound grid velocities
    :param nu_b: Slices for bound upwash
    :param nu_w: Slices for wake upwash
    """

    zeta_b: _LinearComponent
    zeta_b_dot: _LinearComponent
    nu_b: _LinearComponent
    nu_w: _LinearComponent


@dataclass
class StateSlices:
    r"""
    Data class to hold linear components for state components
    :param gamma_b: Slices for bound circulation strengths
    :param gamma_w: Slices for wake circulation strengths
    :param gamma_bm1: Slices for previous time step bound circulation strengths
    :param gamma_b_dot: Slices for bound circulation time derivatives
    :param zeta_w: Slices for wake grid coordinates
    :param zeta_b: Slices for bound grid coordinates
    """

    gamma_b: _LinearComponent
    gamma_w: _LinearComponent
    gamma_bm1: _LinearComponent
    gamma_b_dot: _LinearComponent
    zeta_w: _LinearComponent
    zeta_b: _LinearComponent


@dataclass
class OutputSlices:
    r"""
    Data class to hold linear components for output components
    :param f_steady: Slices for steady force contributions
    :param f_unsteady: Slices for unsteady force contributions
    """

    f_steady: _LinearComponent
    f_unsteady: _LinearComponent


@dataclass
class InputUnflattened:
    r"""
    Data class to hold unflattened input components, for either a single snapshot or a time series.
    :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    :param zeta_b_dot: Bound grid velocities, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    :param nu_b: Bound upwash, [n_surf][m, varphi, 3] or [n_surf][n_ts, m, varphi, 3]
    :param nu_w: Wake upwash, [n_surf][m_star, varphi, 3] or [n_surf][n_ts, m_star, varphi, 3]
    """

    zeta_b: ArrayList
    zeta_b_dot: ArrayList
    nu_b: Optional[ArrayList]
    nu_w: Optional[ArrayList]


@dataclass
class StateUnflattened:
    r"""
    Data class to hold unflattened state components, for either a single snapshot or a time series.
    :param gamma_b: Bound circulation strengths, [n_surf][m, varphi] or [n_surf][n_ts, m, varphi]
    :param gamma_w: Wake circulation strengths, [n_surf][m_star, varphi] or [n_surf][n_ts, m_star, varphi]
    :param gamma_bm1: Previous time step bound circulation strengths, [n_surf][m, varphi] or [n_surf][n_ts, m, varphi]
    :param gamma_b_dot: Bound circulation time derivatives, [n_surf][m, varphi] or [n_surf][n_ts, m, varphi]
    :param zeta_w: Wake grid coordinates, [n_surf][zeta_m_star, zeta_n, 3] or [n_surf][n_ts, zeta_m_star, zeta_n, 3]
    :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    """

    gamma_b: ArrayList
    gamma_w: ArrayList
    gamma_bm1: Optional[ArrayList]
    gamma_b_dot: Optional[ArrayList]
    zeta_w: Optional[ArrayList]
    zeta_b: Optional[ArrayList]


@dataclass
class OutputUnflattened:
    r"""
    Data class to hold unflattened output components, for either a single snapshot or a time series.
    :param f_steady: Steady force contributions, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    :param f_unsteady: Unsteady force contributions, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    """

    f_steady: ArrayList
    f_unsteady: Optional[ArrayList]


class AeroLinearResult:
    def __init__(
            self,
            reference: AeroSnapshot,
            u_t: InputUnflattened,
            x_t: StateUnflattened,
            y_t: OutputUnflattened,
            u_t_tot: InputUnflattened,
            x_t_tot: StateUnflattened,
            y_t_tot: OutputUnflattened,
            n_tstep: int,
            n_surf: int,
            t: Array,
            surf_b_names: list[str],
            surf_w_names: list[str],
    ) -> None:
        # system results, if simulated
        self.u_t: InputUnflattened = u_t
        self.x_t: StateUnflattened = x_t
        self.y_t: OutputUnflattened = y_t
        self.u_t_tot: InputUnflattened = u_t_tot
        self.x_t_tot: StateUnflattened = x_t_tot
        self.y_t_tot: OutputUnflattened = y_t_tot
        self.n_tstep: int = n_tstep
        self.n_surf: int = n_surf
        self.t: Array = t
        self.surf_b_names: list[str] = surf_b_names
        self.surf_w_names: list[str] = surf_w_names
        self.reference: AeroSnapshot = reference

    def plot(
            self,
            directory: str | os.PathLike,
            index: Optional[slice | Sequence[int] | int | Array] = None,
            plot_wake: bool = True,
    ) -> None:
        r"""
        Plot the aerodynamic grid at specified time steps.
        :param directory: Directory to save the plots to
        :param index: Index or slice of time steps to plot. If None, plot all time steps.
        :param plot_wake: If True, plot the wake grid
        """
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif isinstance(index, int):
            index_ = (index,)
        elif index is None:
            index_ = jnp.arange(self.n_tstep)
        else:
            raise TypeError("index must be a slices, sequence of ints, or Array")

        directory_path = Path(directory).resolve()
        directory_path.mkdir(parents=True, exist_ok=True)

        paths: list[Sequence[Path]] = []
        for i_ts in index_:
            snapshot = self[i_ts]
            paths.append(snapshot.plot(directory, plot_wake=plot_wake))

        for i_surf in range(2 * self.n_surf):
            try:
                surf_paths = [paths[i][i_surf] for i in range(len(index_))]
                name = (self.surf_b_names + self.surf_w_names)[i_surf] + "_ts"
                write_pvd(directory, name, surf_paths, list(self.t[index_]))
            except IndexError:
                pass

    def __getitem__(self, i_ts: int) -> AeroSnapshot:
        r"""
        Get snapshot of aerodynamic surface at a single time step
        :param i_ts: Timestep index
        :return: DynamicAeroCase at specified time step
        """

        if i_ts < 0 or i_ts >= self.n_tstep:
            raise IndexError("Timestep index out of range")

        # always exist
        zeta_b_tot = self.u_t_tot.zeta_b.index_all(i_ts, ...)
        zeta_b_dot_tot = self.u_t_tot.zeta_b_dot.index_all(i_ts, ...)
        gamma_b_tot = self.x_t_tot.gamma_b.index_all(i_ts, ...)
        gamma_w_tot = self.x_t_tot.gamma_w.index_all(i_ts, ...)
        f_steady_tot = self.y_t_tot.f_steady.index_all(i_ts, ...)

        # optional
        gamma_b_dot_tot = (
            self.x_t_tot.gamma_b_dot.index_all(i_ts, ...)
            if self.x_t_tot.gamma_b_dot is not None
            else None
        )
        zeta_w_tot = (
            self.x_t_tot.zeta_w.index_all(i_ts, ...)
            if self.x_t_tot.zeta_w is not None
            else None
        )
        f_unsteady_tot = (
            self.y_t_tot.f_unsteady.index_all(i_ts, ...)
            if self.y_t_tot.f_unsteady is not None
            else None
        )

        return AeroSnapshot(
            zeta_b=zeta_b_tot,
            zeta_b_dot=zeta_b_dot_tot,
            zeta_w=zeta_w_tot,
            gamma_b=gamma_b_tot,
            gamma_b_dot=gamma_b_dot_tot,
            gamma_w=gamma_w_tot,
            f_steady=f_steady_tot,
            f_unsteady=f_unsteady_tot,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=self.t,
            horseshoe=False,
            c=None,
            nc=None,
            kernels=self.reference.kernels,
            mirror_point=None,
            mirror_normal=None,
            flowfield=self.reference.flowfield,
            dof_mapping=self.reference.dof_mapping,
        )
