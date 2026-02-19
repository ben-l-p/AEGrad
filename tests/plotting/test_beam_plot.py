from pathlib import Path

from jax import numpy as jnp
import jax

from models.geradin_beam import geradin_beam

jax.config.update("jax_enable_x64", True)


class TestBeamPlot:
    r"""
    Ensures no errors are thrown during plotting of a beam structure.
    """

    @staticmethod
    def test_plot_geradin():
        try:
            m_bar = 1.0
            m_cs = jnp.zeros((6, 6)).at[:3, :3].set(jnp.eye(3) * m_bar)

            struct = geradin_beam(20, "x", m_cs)

            load = 600000.0
            f_ext = jnp.zeros((struct.n_nodes, 6))
            f_ext = f_ext.at[-1, 2].set(-load)

            result = struct.static_solve(
                f_ext,
                f_ext,
                jnp.arange(6),
                load_steps=3,
            )

            plot_dir = Path("./test_beam_plot_output")
            out_file = result.plot(plot_dir).resolve()
            out_file.unlink()  # clean up the generated file after plotting
            plot_dir.rmdir()  # remove the created directory

        except Exception as e:
            assert False, f"Plotting Geradin beam structure failed with error: {e}"
