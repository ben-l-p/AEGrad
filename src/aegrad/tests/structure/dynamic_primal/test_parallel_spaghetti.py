from aegrad.models.flying_spaghetti.flying_spaghetti import generate_flying_spaghetti

import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestParallelSpaghetti:
    r"""
    Simulate many flying spaghetti dynamics problems at once to ensure multi-case structural parallelism works.
    """

    @staticmethod
    def test_parallel_spaghetti_2d():
        n_case = 8
        mult = jnp.linspace(1.0, 1.1, n_case)
        dt = 0.01
        n_tstep = 100
        t = jnp.arange(n_tstep) * dt

        cases = []
        for m in mult:
            case, f_2d, _ = generate_flying_spaghetti(n_nodes=20, t=t)
            case.k_cs *= m
            cases.append(case)

        stacked_case = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *cases)

        def solve(case_):
            return case_.dynamic_solve(
                init_state=None,
                n_tstep=n_tstep,
                dt=dt,
                f_ext_follower=None,
                f_ext_dead=f_2d,
                f_ext_aero=None,
                prescribed_dofs=None,
            )

        solutions = jax.vmap(solve)(stacked_case)
        jax.block_until_ready(solutions)
