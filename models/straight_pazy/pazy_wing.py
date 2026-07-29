from typing import Optional

import jax
from jax import numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation

from aegrad.aero.data_structures import GridDiscretization
from aegrad.aero.flowfields import FlowField, Constant
from aegrad.aero.utils import make_rectangular_grid
from aegrad.aero.uvlm import UVLM
from aegrad.coupled import CoupledAeroelastic
from aegrad.algebra.so3 import vec_to_skew
from aegrad.structure import BeamStructure
from aegrad.utils.data_structures import ConvergenceSettings

# constant from provided data
from models.straight_pazy.pazy_properties import NO_SKIN, WITH_SKIN


def make_generic_pazy_wing(
    data: dict[str, Array],
    m: int,
    m_star: int,
    node_multiplier: int,
    gravity: bool | Array,
    flowfield: FlowField,
    aoa: float | Array,
    sweep: Optional[float | Array],
) -> CoupledAeroelastic:

    # keypoint 3D coordinates at which cross-section properties are defined
    coords_data = jnp.array(data["coords"])  # [N_KEYPOINT, 3]

    # arclength along the beam at each keypoint, used as the interpolation parameter
    seg_lengths_data = jnp.linalg.norm(jnp.diff(coords_data, axis=0), axis=1)
    s_data = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths_data)])

    # coordinates we wish to use, for which we will interpolate properties
    coords = coords_data
    if node_multiplier > 1:
        # subdivide each keypoint segment into node_multiplier equal parts in 3D
        frac = jnp.linspace(0.0, 1.0, node_multiplier + 1)[:-1]
        sub_nodes = (
            coords[:-1, None, :]
            + (coords[1:, None, :] - coords[:-1, None, :]) * frac[None, :, None]
        )
        coords = jnp.concatenate([sub_nodes.reshape(-1, 3), coords[-1:]], axis=0)
    n_nodes = coords.shape[0]
    n_elem = n_nodes - 1

    # arclength midpoints of the AEGrad elements, used for property interpolation
    seg_lengths = jnp.linalg.norm(jnp.diff(coords, axis=0), axis=1)
    s_arr = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths)])
    elem_midpoints = 0.5 * (s_arr[:-1] + s_arr[1:])

    cg_data = jnp.stack(
        (data["cg_y"], -data["cg_x"], data["cg_z"]), axis=1
    )  # [N_KEYPOINT, 3]

    # inertia tensor at CG, in beam-local frame
    j_cg_data = jnp.zeros((data["n_points"], 3, 3))

    idx_keys = (
        (0, 0, "i_yy"),
        (1, 1, "i_xx"),
        (2, 2, "i_zz"),
        (0, 1, "i_xy"),
        (1, 2, "i_xz"),
        (0, 2, "i_yz"),
    )
    for i, j, key in idx_keys:
        # redundant for diagonal terms
        j_cg_data = j_cg_data.at[:, i, j].set(data[key])
        j_cg_data = j_cg_data.at[:, j, i].set(data[key])

    # parallel-axis shift inertia
    skew_cg_data = jax.vmap(vec_to_skew)(cg_data)  # [N_KEYPOINT, 3]
    j_axis_data = j_cg_data - data["mass"][:, None, None] * (
        skew_cg_data @ skew_cg_data
    )

    # distribute nodal lumped mass and inertia
    elem_lengths_data = seg_lengths_data  # [N_KEYPOINT - 1]
    half_lengths_data = jnp.concatenate(
        (
            0.5 * elem_lengths_data[:1],
            0.5 * (elem_lengths_data[:-1] + elem_lengths_data[1:]),
            0.5 * elem_lengths_data[-1:],
        )
    )  # [N_KEYPOINT]
    m_bar_data = data["mass"] / half_lengths_data  # mass per unit length of elements
    j_bar_data = (
        j_axis_data / half_lengths_data[:, None, None]
    )  # inertia per unit length of elements

    # interpolate properties from the arclength keypoints
    m_bar = jnp.interp(elem_midpoints, s_data, m_bar_data)
    cg_elem = jnp.stack(
        [jnp.interp(elem_midpoints, s_data, cg_data[:, i]) for i in range(3)],
        axis=1,
    )
    j_bar = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.interp(elem_midpoints, s_data, j_bar_data[:, i, k])
                    for k in range(3)
                ],
                axis=1,
            )
            for i in range(3)
        ],
        axis=1,
    )  # [n_elem, 3, 3]

    # assemble the 6x6 cross-section mass matrix per beam element
    skew_cg_elem = jax.vmap(vec_to_skew)(cg_elem)
    m_bar_ = m_bar[:, None, None]
    m_cs = jnp.zeros((n_elem, 6, 6))
    m_cs = m_cs.at[:, :3, :3].set(m_bar_ * jnp.eye(3))
    m_cs = m_cs.at[:, :3, 3:].set(-m_bar_ * skew_cg_elem)
    m_cs = m_cs.at[:, 3:, :3].set(m_bar_ * skew_cg_elem)
    m_cs = m_cs.at[:, 3:, 3:].set(j_bar)

    # stiffness properties
    # data is provided for non-shear deformation
    k_cs_data = jnp.zeros((data["n_points"] - 1, 6, 6))

    idx_keys = (
        (0, 0, "k11"),
        (3, 3, "k22"),
        (4, 4, "k33"),
        (5, 5, "k44"),
        (0, 3, "k12"),
        (0, 4, "k13"),
        (0, 5, "k14"),
        (3, 4, "k23"),
        (3, 5, "k24"),
        (4, 5, "k34"),
    )
    for i, j, key in idx_keys:
        # redundant for diagonal terms
        k_cs_data = k_cs_data.at[:, i, j].set(data[key])
        k_cs_data = k_cs_data.at[:, j, i].set(data[key])

    # add large shear stiffness to diagonal terms to avoid singularity
    k_cs_data = k_cs_data.at[:, 1, 1].set(1e9)
    k_cs_data = k_cs_data.at[:, 2, 2].set(1e9)
    data_elem_midpoints = 0.5 * (
        s_data[:-1] + s_data[1:]
    )  # arclength midpoints of the data segments, used for interpolation

    # interpolate each stiffness component at the new element midpoints;
    k_cs = interp_cs_property(
        data=k_cs_data, data_coords=data_elem_midpoints, coords=elem_midpoints
    )

    # aerodynamic model
    c_ref = 0.1
    ea = 0.441
    aero_grid = make_rectangular_grid(m=m, n=n_nodes - 1, chord=c_ref, ea=ea)

    # make wing
    conn = jnp.stack(
        [jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)], axis=1
    )  # [n_elem, 2]
    y_vector = jnp.zeros((n_elem, 3))  # [n_elem, 3]
    y_vector = y_vector.at[:, 0].set(-1.0)

    dt = c_ref / (flowfield.u_inf_mag * m)  # time step based on CFL condition

    if isinstance(gravity, Array):
        gravity_ = gravity
    elif gravity:
        gravity_ = jnp.array((0.0, -9.81, 0.0))
    else:
        gravity_ = None

    if sweep is not None:
        # apply sweep as a postprocessing step to rotate the coordinates
        rmat = Rotation.from_euler("zyx", jnp.array((-sweep, 0.0, 0.0))).as_matrix()
        sweep_coords = jnp.einsum("ij, kj->ki", rmat, coords)
    else:
        sweep_coords = coords

    structure = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=y_vector,
        k_cs_index=jnp.arange(n_elem),
        m_cs_index=jnp.arange(n_elem),
        gravity=gravity_,
    )
    structure.struct_convergence_settings = ConvergenceSettings(
        max_n_iter=100,
        rel_disp_tol=1e-8,
        abs_disp_tol=1e-8,
        rel_force_tol=1e-5,
        abs_force_tol=1e-3,
    )

    aero = UVLM(
        grid_shapes=(GridDiscretization(m=m, n=n_nodes - 1, m_star=m_star),),
        dof_mapping=jnp.arange(n_nodes),
        mirror_point=jnp.zeros(3),
        mirror_normal=jnp.array((0.0, 1.0, 0.0)),
    )
    wing = CoupledAeroelastic(structure=structure, aero=aero)
    wing.fsi_convergence_settings = ConvergenceSettings(
        max_n_iter=40,
        rel_disp_tol=1e-3,
        abs_disp_tol=1e-3,
        rel_force_tol=1e-2,
        abs_force_tol=1e-2,
    )

    wing.set_design_variables(
        coords=sweep_coords,
        k_cs=k_cs,
        m_cs=m_cs,
        m_lumped=None,
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        x0_aero=aero_grid,
        orientation_euler=jnp.array((0.0, aoa, 0.0)),
    )

    return wing


def interp_cs_property(data: Array, data_coords: Array, coords: Array) -> Array:
    return jnp.stack(
        [
            jnp.stack(
                [jnp.interp(coords, data_coords, data[:, i, j]) for j in range(6)],
                axis=1,
            )
            for i in range(6)
        ],
        axis=1,
    )


def make_pazy_wing(
    m: int = 12,
    m_star: int = 120,
    skin: bool = True,
    node_multiplier: int = 1,
    gravity: bool | Array = False,
    flowfield: FlowField = Constant(
        u_inf=jnp.array((40.0, 0.0, 0.0)), rho=1.225, relative_motion=True
    ),
    aoa: float | Array = jnp.deg2rad(3.0),
    sweep: Optional[float | Array] = None,
) -> CoupledAeroelastic:
    return make_generic_pazy_wing(
        m=m,
        m_star=m_star,
        node_multiplier=node_multiplier,
        gravity=gravity,
        flowfield=flowfield,
        aoa=aoa,
        data=WITH_SKIN if skin else NO_SKIN,
        sweep=sweep,
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    rho = 1.225
    aoa_ = jnp.deg2rad(7.0)
    u_inf_mag = 60.0

    wing_ = make_pazy_wing(
        gravity=True,
        node_multiplier=2,
        m=16,
        m_star=160,
        skin=True,
        aoa=aoa_,
        flowfield=Constant(
            rho=rho, u_inf=jnp.array((u_inf_mag, 0.0, 0.0)), relative_motion=True
        ),
        sweep=jnp.deg2rad(10.0),
    )
    static_sol = wing_.static_solve(
        prescribed_dofs=jnp.arange(6), horseshoe=False, load_steps=1, fsi_relaxation=0.5
    )
    static_sol.plot("./pazy_outputs/")

    z_tip = 0.5 * (
        static_sol.aero.zeta_b[0][0, -1, 2] + static_sol.aero.zeta_b[0][-1, -1, 2]
    )
    print(f"Tip deflection at mid chord (m): {float(z_tip):.03f}")
    print(f"Relative tip deflection at mid chord (z/b) {float(z_tip) / 0.55:.03f}")
