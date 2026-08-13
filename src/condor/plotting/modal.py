from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from jax import Array, vmap
from jax import numpy as jnp

from condor.aero.data_structures import AeroSnapshot
from condor.algebra.array_utils import ArrayList
from condor.algebra.se3 import exp_se3
from condor.plotting.aerogrid import plot_grid_to_vtk
from condor.plotting.beam import plot_beam_to_vtk
from condor.plotting.pvd import write_pvd
from condor.structure import StaticStructure

if TYPE_CHECKING:
    from condor.aero.uvlm import UVLM
    from condor.coupled import StaticAeroelastic


def plot_modes_vtu(
    reference: StaticAeroelastic | StaticStructure,
    q_full: Array,
    freqs: Array,
    dampings: Array,
    gamma_b_full: ArrayList | None,
    gamma_w_full: ArrayList | None,
    zeta_w_full: ArrayList | None,
    uvlm: UVLM | None,
    directory: os.PathLike | str = "./modal",
    n_phase: int = 8,
    n_interp: int = 0,
    max_disp: float = 0.2,
    max_ang: float = 0.2,
    max_gamma: float = 100.0,
) -> None:
    r"""
    Plot aeroelastic modes for visualisation in paraview.
    :param reference: Reference aeroelastic model used for linearisation.
    :param directory: Directory for saving data. If not passed, will default to ./modal.
    :param q_full: Full nodal mode deflections, (n_modes, n_nodes, 6).
    :param freqs: Damped mode frequencies, Hz. (n_modes), used for file naming.
    :param dampings: Damping ratios, (n_modes), used for file naming.
    :param gamma_b_full: Full bound gamma modes, (n_surf)(n_modes, m, n) or None.
    :param gamma_w_full: Full wake modes, (n_surf)(n_modes, m*, n) or None.
    :param zeta_w_full: Full wake displacement modes, (n_surf)(n_modes, m_star, n, 3) or None.
    :param uvlm: UVLM model, used for generating the aerodynamic grid.
    :param n_phase: Number of phase points to plot the modes.
    :param n_interp: Number of interpolation points for plotting the structure.
    :param max_disp: Maximum nodal displacement for mode normalisation.
    :param max_ang: Maximum nodal angle for mode normalisation.
    :param max_gamma: Maximum gamma for mode normalisation.
    """

    if (gamma_b_full is None and gamma_w_full is not None) or (
        gamma_w_full is None and gamma_b_full is not None
    ):
        raise ValueError(
            "Both gamma_b and gamma_w must simultaneously be None or not None."
        )

    if gamma_b_full is not None and gamma_w_full is not None and uvlm is None:
        raise ValueError("UVLM model must be provided when plotting aeroelastic modes.")

    phase_circle = jnp.exp(
        1j * jnp.linspace(0.0, 2.0 * jnp.pi, n_phase, endpoint=False)
    )

    plot_dir = Path(directory)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(reference, StaticStructure):
        struct_ref: StaticStructure = reference
        aero_ref: AeroSnapshot | None = None
    else:
        struct_ref: StaticStructure = reference.structure
        aero_ref: AeroSnapshot | None = reference.aero

    conn = jnp.array(struct_ref.conn, dtype=int)
    o0 = struct_ref.o0

    n_modes = q_full.shape[0]
    for i_mode in range(n_modes):
        eig_max_disp = jnp.linalg.norm(q_full[i_mode, :, :3], axis=1).max()
        eig_max_ang = jnp.linalg.norm(q_full[i_mode, :, 3:], axis=1).max()

        eig_max_gamma = (
            jnp.maximum(
                jnp.abs(gamma_b_full.index_all(i_mode, ...).ravel()).max(),
                jnp.abs(gamma_w_full.index_all(i_mode, ...).ravel()).max()
                if gamma_w_full.size
                else 0.0,
            )
            if gamma_b_full is not None and gamma_w_full is not None
            else 1e-9
        )

        # scale to whichever limit is reached first; ignore vanishing components
        scale_factor = jnp.array(
            (
                jnp.where(eig_max_disp > 0, max_disp / eig_max_disp, jnp.inf),
                jnp.where(eig_max_ang > 0, max_ang / eig_max_ang, jnp.inf),
                jnp.where(eig_max_gamma > 0, max_gamma / eig_max_gamma, jnp.inf),
            )
        ).min()

        # rescale
        delta_q_full: Array = q_full[i_mode, ...] * scale_factor

        delta_gamma_b: ArrayList | None = (
            gamma_b_full.index_all(i_mode, ...) * scale_factor
            if gamma_b_full is not None
            else None
        )
        delta_gamma_w: ArrayList | None = (
            gamma_w_full.index_all(i_mode, ...) * scale_factor
            if gamma_w_full is not None
            else None
        )
        delta_zeta_w: ArrayList | None = (
            zeta_w_full.index_all(i_mode, ...) * scale_factor
            if zeta_w_full is not None
            else None
        )

        mode_dir = plot_dir.joinpath(
            f"mode_{i_mode + 1}_f{float(freqs[i_mode]):.2f}Hz_z{float(dampings[i_mode]):+.4f}"
        )
        mode_dir.mkdir(parents=True, exist_ok=True)
        beam_paths: list[Path] = []

        if uvlm is not None:
            n_surf = uvlm.n_surf
            bound_paths: list[list[Path]] = [[] for _ in range(n_surf)]
            wake_paths: list[list[Path]] = [[] for _ in range(n_surf)]
        else:
            n_surf = 0
            bound_paths = []
            wake_paths = []

        for i_phase, phase in enumerate(phase_circle):
            # real perturbation at this phase, added to the reference state
            q_pert_full: Array = (delta_q_full * phase).real  # (n_nodes, 6)
            gamma_b: ArrayList | None = (
                (
                    ArrayList([(a * phase).real for a in delta_gamma_b])
                    + aero_ref.gamma_b
                )
                if delta_gamma_b is not None and aero_ref is not None
                else None
            )
            gamma_w: ArrayList | None = (
                (
                    ArrayList([(a * phase).real for a in delta_gamma_w])
                    + aero_ref.gamma_w
                )
                if delta_gamma_w is not None and aero_ref is not None
                else None
            )
            zeta_w: ArrayList | None = (
                ArrayList([(a * phase).real for a in delta_zeta_w]) + aero_ref.zeta_w
                if delta_zeta_w is not None and aero_ref is not None
                else aero_ref.zeta_w
                if aero_ref is not None
                else None
            )

            hg = jnp.einsum(
                "ijk,ikl->ijl",
                struct_ref.hg,
                vmap(exp_se3)(q_pert_full),
            )

            beam_paths.append(
                plot_beam_to_vtk(
                    hg=hg,
                    conn=conn,
                    o0=o0,
                    n_interp=n_interp,
                    filename=mode_dir.joinpath("beam"),
                    i_ts=i_phase,
                )
            )

            if uvlm is not None:
                # plot aerodynamic grid
                zeta_b: ArrayList = uvlm.hg_to_zeta_b(hg_n=hg, cs_ang_n=uvlm.cs_ang0)

                for i_surf in range(n_surf):
                    bound_paths[i_surf].append(
                        plot_grid_to_vtk(
                            grid_arr=zeta_b[i_surf],
                            filename=mode_dir.joinpath(uvlm.surf_b_names[i_surf]),
                            i_ts=i_phase,
                            cell_scalar_data={"gamma": gamma_b[i_surf]}
                            if gamma_b is not None
                            else None,
                        )
                    )

                    if gamma_w is not None:
                        assert zeta_w is not None
                        if zeta_w[i_surf].shape[0] > 1:
                            wake_paths[i_surf].append(
                                plot_grid_to_vtk(
                                    grid_arr=zeta_w[i_surf],
                                    filename=mode_dir.joinpath(
                                        uvlm.surf_w_names[i_surf]
                                    ),
                                    i_ts=i_phase,
                                    cell_scalar_data={"gamma": gamma_w[i_surf]},
                                )
                            )

        times = (
            jnp.linspace(0.0, 1.0 / freqs[i_mode], n_phase, endpoint=False)
        ).tolist()

        # link timesteps with PVD
        write_pvd(
            directory=mode_dir, name="beam", file_dirs=beam_paths, times=times
        )  # always write beam data

        for i_surf in range(n_surf):
            # write for aero grid data if it exists
            if bound_paths[i_surf] and uvlm is not None:
                write_pvd(
                    directory=mode_dir,
                    name=uvlm.surf_b_names[i_surf],
                    file_dirs=bound_paths[i_surf],
                    times=times,
                )
            if wake_paths[i_surf] and uvlm is not None:
                write_pvd(
                    directory=mode_dir,
                    name=uvlm.surf_w_names[i_surf],
                    file_dirs=wake_paths[i_surf],
                    times=times,
                )
