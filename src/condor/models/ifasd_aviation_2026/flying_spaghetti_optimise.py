from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from condor.models.flying_spaghetti.flying_spaghetti import generate_flying_spaghetti
from condor.structure import StructuralDesignVariables, StructureFullStates
from condor.structure.gradients.data_structures import StructuralGradsToCompute

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    r"""
    Optimize the per-element cross-section mass per unit length (m_bar) of the flying spaghetti to keep the spaghetti
    as straight as possible at the end of the simulation. This is subject to the same force and moment at the root as
    in the reference case, with the stiffness of the model scaled down by 3 times. 
    """
    n_nodes_ = 21
    n_elem_ = n_nodes_ - 1
    dt_ = 0.01
    t_end_ = 10.0
    n_tstep_ = int(jnp.ceil(t_end_ / dt_)) + 1
    n_iter = 50
    m_min = 0.3
    m_max = 3.0

    # start one timestep behind to prevent issues with initial condition
    t_ = jnp.arange(n_tstep_) * dt_ - dt_

    dir_ = Path("./flying_spaghetti_optimise/")

    # objective which refers to a single timestep
    def objective(
        states: StructureFullStates,
        _: StructuralDesignVariables,
        i_ts: int,
    ) -> Array:
        return jax.lax.select(
            i_ts == n_tstep_ - 1, (states.eps**2).sum(), 0.0
        )  # inner product of strains as a measure for how "straight" the spaghetti is

    iter_counter = {"i": 0}  # prevents needing global or nonlocal keywords in make_sol

    def make_sol(m_bar: np.ndarray) -> tuple[float, Array]:
        # inner loop, which takes a mass distribution and returns the objective value and its gradient
        i_iter = iter_counter["i"]
        iter_counter["i"] += 1

        m_bar_j = jnp.asarray(m_bar)

        struct_, f_dead_2d_, _ = generate_flying_spaghetti(
            n_nodes_,
            t_,
            spectral_radius=0.7,
            m_bar=m_bar_j,
        )

        # reduce stiffness to make the problem more interesting
        struct_.k_cs /= 3.0

        primal_sol = struct_.dynamic_solve(
            init_state=None,
            n_tstep=n_tstep_,
            dt=dt_,
            f_ext_follower=None,
            f_ext_dead=f_dead_2d_,
            f_ext_aero=None,
            prescribed_dofs=None,
        )

        grads, _ = struct_.dynamic_adjoint(
            structure=primal_sol,
            objective=objective,
            p_q0_p_x=None,
            approx_grads=True,
            grads_to_compute=StructuralGradsToCompute(
                x0=False,
                k_cs=False,
                m_cs=True,
                m_lumped=False,
                f_ext_follower=False,
                f_ext_dead=False,
            ),
        )

        assert grads.m_cs is not None
        # grads.m_cs has shape (n_obj=1, n_elem, 6, 6); m_bar feeds the (0,0), (1,1), (2,2) diagonals per element
        m_bar_grad = (
            grads.m_cs[0, :, 0, 0] + grads.m_cs[0, :, 1, 1] + grads.m_cs[0, :, 2, 2]
        )

        objective_val = (primal_sol.eps[-1, ...] ** 2).sum()

        iter_path = dir_.joinpath(f"iteration_{i_iter}/")
        iter_path.mkdir(parents=True, exist_ok=True)

        # plot the beam time history VTK
        primal_sol.plot(iter_path.joinpath("vtk"))

        # snapshot of beam at end of simulation
        _, ax = plt.subplots()
        ax.set_title(f"Beam at {i_iter}, objective = {float(objective_val)}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.axis("equal")
        ax.plot(primal_sol.hg[-1, :, 0, 3], primal_sol.hg[-1, :, 2, 3])
        plt.savefig(iter_path.joinpath("end_coords.png"))

        # mass distribution
        _, ax = plt.subplots()
        ax.plot(jnp.arange(n_elem_), m_bar_j)
        ax.set_title(f"Element m_bar at {i_iter}")
        ax.set_xlabel("Element number")
        ax.set_ylabel("m_bar, kg/m")
        plt.savefig(iter_path.joinpath("m_bar.png"))

        # mass distribution gradient
        _, ax = plt.subplots()
        ax.plot(jnp.arange(n_elem_), m_bar_grad)
        ax.set_title(f"Gradient at {i_iter}")
        ax.set_xlabel("Element number")
        ax.set_ylabel("d(obj) / d(m_bar)")
        plt.savefig(iter_path.joinpath("m_bar_grad.png"))

        plt.close("all")

        print(f"iteration {i_iter}: objective = {float(objective_val):.6e}")

        return float(objective_val), m_bar_grad

    # initial guess as uniform mass per unit length of 1.0
    m_bar0 = np.ones(n_elem_)

    # constrain the total mass to be constant
    total_mass_target = float(m_bar0.sum())
    constraints = (
        {
            "type": "eq",
            "fun": lambda x: x.sum() - total_mass_target,
            "jac": lambda x: np.ones_like(x),
        },
    )

    # optimise the mass distribution
    # noinspection SpellCheckingInspection
    result = minimize(  # type: ignore
        fun=make_sol,
        x0=m_bar0,
        jac=True,
        bounds=[(m_min, m_max)] * n_elem_,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": n_iter, "ftol": 1e-8},
    )
