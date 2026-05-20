import jax

from aegrad.aero.data_structures import GridDiscretization
from aegrad.aero.flowfields import FlowField, Constant
from aegrad.aero.utils import make_rectangular_grid
from aegrad.aero.uvlm import UVLM
from aegrad.coupled import CoupledAeroelastic

from jax import numpy as jnp
from jax import Array

from structure import BeamStructure


def simple_hale(
    sigma_wing: float = 1.5,
    sigma_fuselage: float = 10.0,
    sigma_tail: float = 100.0,
    flowfield: FlowField = Constant(
        rho=1.225,
        u_inf=jnp.array((10.0, 0.0, 0.0)),
        relative_motion=True,
    ),
) -> CoupledAeroelastic:
    r"""
    Generate the simple HALE aircraft. Reference parameters are derived from the SHARPy case.
    :param sigma_wing: Stiffness multiplier for the wing structure.
    :param sigma_fuselage: Stiffness multiplier for the fuselage structure.
    :param sigma_tail: Stiffness multiplier for the tail structure.
    :param flowfield: FlowField object.
    :return: HALE aircraft object.
    """
    # geometry
    c_wing: float = 1.0
    c_tail: float = 0.5
    b_tail: float = 2.5
    b_wing: float = 16.0
    l_fuselage: float = 10.0
    height_tail: float = 2.5
    elastic_axis_wing: float = 0.3
    elastic_axis_tail: float = 0.5
    dihedral_ang: Array = jnp.deg2rad(20.0)
    dihedral_fraction: float = 0.25

    # structural properties
    m_payload: float = 50.0
    m_bar_wing: float = 0.75
    m_bar_tail: float = 0.08
    m_bar_fuselage: float = 0.2
    j_bar_wing: float = 0.075
    j_bar_tail: float = 0.01
    j_bar_fuselage: float = 0.08
    gj: float = 1e4
    ea: float = 1e7
    ga: float = 1e5
    eiy = 2e4  # out-of-plane
    eiz = 4e6  # in-plane

    # discretisation
    n_half_wing: int = 16
    n_outer_wing: int = int(n_half_wing * dihedral_fraction)
    n_inner_wing: int = n_half_wing - n_outer_wing
    m_wing: int = 4
    n_fuselage: int = 8
    n_fin: int = 4
    n_half_tail: int = 4
    m_tail: int = 4
    m_star: int = 20

    # other variables
    dt: float = c_wing / (m_wing * float(jnp.linalg.norm(flowfield.u_inf)))
    thrust_reference: float = 6.16
    alpha: Array = jnp.deg2rad(4.31)
    roll: float = 0.0
    beta: float = 0.0

    # useful reference coordinates
    left_wing_junction_coord = jnp.array((0.0, (1.0 - dihedral_fraction) * b_wing, 0.0))
    left_wing_tip_coord = left_wing_junction_coord + jnp.array(
        (
            0.0,
            dihedral_fraction * b_wing * jnp.cos(dihedral_ang),
            dihedral_fraction * b_wing * jnp.sin(dihedral_ang),
        )
    )
    tail_base_coord = jnp.array((l_fuselage, 0.0, 0.0))
    tail_root_coord = jnp.array((l_fuselage, 0.0, height_tail))
    left_tail_tip_coord = tail_root_coord + jnp.array((0.0, b_tail, 0.0))

    # create beam coordinates
    left_wing_coords: Array = jnp.zeros((n_half_wing + 1, 3))  # includes zero node
    left_wing_coords = left_wing_coords.at[
        : n_inner_wing + 1, :
    ].set(  # inner straight wing
        jnp.outer(jnp.linspace(0.0, 1.0, n_inner_wing + 1), left_wing_junction_coord)
    )
    left_wing_coords = left_wing_coords.at[n_inner_wing:, :].set(  # add outer dihedral
        jnp.outer(
            jnp.linspace(0.0, 1.0, n_outer_wing + 1),
            left_wing_tip_coord - left_wing_junction_coord,
        )
        + left_wing_junction_coord[None, :]
    )

    right_wing_coords: Array = left_wing_coords[1:, :].at[:, 1].mul(-1.0)  # mirror wing

    fuselage_coords: Array = jnp.outer(
        jnp.linspace(0.0, 1.0, n_fuselage + 1)[1:], tail_base_coord
    )

    fin_coords: Array = tail_base_coord[None, :] + jnp.outer(
        jnp.linspace(0.0, 1.0, n_fin + 1)[1:], (tail_root_coord - tail_base_coord)
    )

    left_tail_coords: Array = tail_root_coord[None, :] + jnp.outer(
        jnp.linspace(0.0, 1.0, n_half_tail + 1)[1:],
        (left_tail_tip_coord - tail_root_coord),
    )

    right_tail_coords: Array = left_tail_coords[...].at[:, 1].mul(-1.0)  # mirror tail

    coords = jnp.concatenate(
        (
            left_wing_coords,
            right_wing_coords,
            fuselage_coords,
            fin_coords,
            left_tail_coords,
            right_tail_coords,
        )
    )

    # index of components in full coordinates
    left_wing_coord_index = jnp.arange(n_half_wing + 1)
    right_wing_coord_index = jnp.array(
        (0, *jnp.arange(n_half_wing) + left_wing_coord_index[-1] + 1)
    )
    fuselage_coord_index = jnp.array(
        (0, *jnp.arange(n_fuselage) + right_wing_coord_index[-1] + 1)
    )
    fin_coord_index = jnp.arange(n_fin + 1) + fuselage_coord_index[-1]
    left_tail_coord_index = jnp.arange(n_half_tail + 1) + fin_coord_index[-1]
    right_tail_coord_index = jnp.array(
        (fin_coord_index[-1], *(jnp.arange(n_fin) + left_tail_coord_index[-1] + 1))
    )

    # connectivity matrices
    conn_left_wing = jnp.stack(
        (left_wing_coord_index[:-1], left_wing_coord_index[1:]), axis=1
    )
    conn_right_wing = jnp.stack(
        (right_wing_coord_index[:-1], right_wing_coord_index[1:]), axis=1
    )
    conn_fuselage = jnp.stack(
        (fuselage_coord_index[:-1], fuselage_coord_index[1:]), axis=1
    )
    conn_fin = jnp.stack((fin_coord_index[:-1], fin_coord_index[1:]), axis=1)
    conn_left_tail = jnp.stack(
        (left_tail_coord_index[:-1], left_tail_coord_index[1:]), axis=1
    )
    conn_right_tail = jnp.stack(
        (right_tail_coord_index[:-1], right_tail_coord_index[1:]), axis=1
    )
    conn = jnp.concatenate(
        (
            conn_left_wing,
            conn_right_wing,
            conn_fuselage,
            conn_fin,
            conn_left_tail,
            conn_right_tail,
        ),
        axis=0,
    )

    wing_element_index = jnp.arange(2 * n_half_wing)
    fuselage_element_index = jnp.arange(n_fuselage) + wing_element_index[-1] + 1
    tail_element_index = (
        jnp.arange(2 * n_half_tail + n_fin) + fuselage_element_index[-1] + 1
    )

    # mass and stiffness matrices
    m_cs_wing = jnp.diag(
        jnp.array(
            (m_bar_wing, m_bar_wing, m_bar_wing, j_bar_wing, j_bar_wing, j_bar_wing)
        )
    )
    m_cs_fuselage = jnp.diag(
        jnp.array(
            (
                m_bar_fuselage,
                m_bar_fuselage,
                m_bar_fuselage,
                j_bar_fuselage,
                j_bar_fuselage,
                j_bar_fuselage,
            )
        )
    )
    m_cs_tail = jnp.diag(
        jnp.array(
            (m_bar_tail, m_bar_tail, m_bar_tail, j_bar_tail, j_bar_tail, j_bar_tail)
        )
    )
    k_cs_wing = jnp.diag(
        jnp.array((ea, ga, ga, gj * sigma_wing, eiy * sigma_wing, eiz * sigma_wing))
    )
    k_cs_fuselage = jnp.diag(
        jnp.array(
            (
                ea,
                ga,
                ga,
                gj * sigma_fuselage,
                eiy * sigma_fuselage,
                eiz * sigma_fuselage,
            )
        )
    )
    k_cs_tail = jnp.diag(
        jnp.array((ea, ga, ga, gj * sigma_tail, eiy * sigma_tail, eiz * sigma_tail))
    )
    m_lump = (
        jnp.zeros((6, 6))
        .at[:3, :3]
        .set(jnp.diag(jnp.array((m_payload, m_payload, m_payload))))
    )[None, :]

    # index of 0 for wing, 1 for fuselage and 2 for tail
    elem_type_index = jnp.concatenate(
        (
            jnp.zeros(2 * n_half_wing, dtype=int),
            jnp.full(n_fuselage, 1, dtype=int),
            jnp.full(n_fin + 2 * n_half_tail, 2, dtype=int),
        )
    )

    # columns are the y vector for the wing, fuselage and tails
    y_vectors = jnp.concatenate(
        (
            jnp.broadcast_to(jnp.array((1.0, 0.0, 0.0)), (2 * n_half_wing, 3)),
            jnp.broadcast_to(jnp.array((0.0, 1.0, 0.0)), (n_fuselage, 3)),
            jnp.broadcast_to(jnp.array((0.0, 1.0, 0.0)), (n_fin, 3)),
            jnp.broadcast_to(jnp.array((1.0, 0.0, 0.0)), (2 * n_half_tail, 3)),
        ),
        axis=0,
    )

    structure = BeamStructure(
        num_nodes=coords.shape[0],
        connectivity=conn,
        y_vector=y_vectors,
        m_cs_index=elem_type_index,
        k_cs_index=elem_type_index,
        m_lumped_index=jnp.array(0, dtype=int),
        gravity=jnp.array((0.0, 0.0, -9.81)),
        thrust_nodes={"thrust": 0},
        thrust_direction={"thrust": jnp.array((1, 0, 0))},
    )

    # aerodynamic_grids
    # wing
    wing_x0 = make_rectangular_grid(
        m=m_wing, n=2 * n_half_wing, chord=c_wing, ea=elastic_axis_wing
    )
    wing_mapping = jnp.concatenate(
        (left_wing_coord_index[::-1], right_wing_coord_index[1:])
    )
    wing_grid_shape = GridDiscretization(m=m_wing, n=2 * n_half_wing, m_star=m_star)

    # fin
    fin_x0 = make_rectangular_grid(
        m=m_tail, n=n_fin, chord=c_tail, ea=elastic_axis_tail
    )
    fin_mapping = fin_coord_index
    fin_grid_shape = GridDiscretization(m=m_tail, n=n_fin, m_star=m_star)

    # tail
    tail_x0 = make_rectangular_grid(
        m=m_tail, n=2 * n_half_tail, chord=c_tail, ea=elastic_axis_tail
    )
    tail_mapping = jnp.concatenate(
        (left_tail_coord_index[::-1], right_tail_coord_index[1:])
    )
    tail_grid_shape = GridDiscretization(m=m_tail, n=2 * n_half_tail, m_star=m_star)

    aero = UVLM(
        grid_shapes=(wing_grid_shape, fin_grid_shape, tail_grid_shape),
        dof_mapping=(wing_mapping, fin_mapping, tail_mapping),
    )
    wing = CoupledAeroelastic(aero=aero, structure=structure)
    wing.set_design_variables(
        coords=coords,
        k_cs=jnp.stack([k_cs_wing, k_cs_fuselage, k_cs_tail], axis=0),
        m_cs=jnp.stack([m_cs_wing, m_cs_fuselage, m_cs_tail], axis=0),
        m_lumped=m_lump,
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        x0_aero=[wing_x0, fin_x0, tail_x0],
        thrust_reference={"thrust": jnp.array(thrust_reference)},
        orientation_euler=jnp.array((roll, alpha, beta)),
    )

    return wing


if __name__ == "__main__":
    # run a clamped case
    jax.config.update("jax_enable_x64", True)
    hale = simple_hale()
    sol = hale.static_solve(prescribed_dofs=jnp.arange(6))
    sol.plot("./simple_hale/")
