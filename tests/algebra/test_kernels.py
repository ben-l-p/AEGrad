from jax import numpy as jnp

from aegrad.aero.kernels import _biot_savart, _biot_savart_epsilon, _biot_savart_cutoff
from aegrad.constants import R_CUTOFF


class TestLinearOperator:
    @staticmethod
    def test_biot_savart():
        r"""
        Test the biot_savart kernel function
        """

        # influence at midpoint should be NaN
        x = jnp.array([0.0, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.all(jnp.isnan(bs := _biot_savart(x, y))), (
            f"Expected NaN, but got {bs}"
        )

        # collinear influnence should be nan
        x = jnp.array([0.0, 1.0, 0.0])
        y = jnp.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.all(jnp.isnan(bs := _biot_savart(x, y))), (
            f"Expected NaN, but got {bs}"
        )

        # defined influnence when not on the filament line
        x = jnp.array([0.1, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        assert not jnp.all(jnp.isnan(bs := _biot_savart(x, y))), (
            f"Expected defined value, but got {bs}"
        )

    @staticmethod
    def test_biot_savart_epsilon():
        r"""
        Test the biot_savart epsillon kernel function
        """

        # influence at midpoint should be zero
        x = jnp.array([0.0, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.allclose(bs := _biot_savart_epsilon(x, y), 0.0), (
            f"Expected zero, but got {bs}"
        )

        # collinear influnence should be zero
        x = jnp.array([0.0, 1.0, 0.0])
        y = jnp.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.allclose(bs := _biot_savart_epsilon(x, y), 0.0), (
            f"Expected zero, but got {bs}"
        )

        # defined influnence when not on the filament line should approximately match baseline
        x = jnp.array([0.1, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        bs = _biot_savart(x, y)
        bse = _biot_savart_epsilon(x, y)
        assert jnp.allclose(bs, bse), (
            f"Returned {bse}, but expected {bs} from baseline kernel"
        )

    @staticmethod
    def test_biot_savart_cutoff():
        r"""
        Test the biot_savart cutoff kernel function
        """

        # influence at midpoint should be NaN
        x = jnp.array([0.0, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.allclose(bs := _biot_savart_cutoff(x, y), 0.0), (
            f"Expected zero, but got {bs}"
        )

        # collinear influence should be zero
        x = jnp.array([0.0, 1.0, 0.0])
        y = jnp.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.allclose(bs := _biot_savart_cutoff(x, y), 0.0), (
            f"Expected zero, but got {bs}"
        )

        # defined influence when not on the filament line should match baseline
        x = jnp.array([1.01 * R_CUTOFF, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        bs = _biot_savart(x, y)
        bsc = _biot_savart_cutoff(x, y)
        assert jnp.allclose(bs, bsc), (
            f"Returned {bsc}, but expected {bs} from baseline kernel"
        )

        # influence within cutoff radius should be zero
        x = jnp.array([0.99 * R_CUTOFF, 0.0, 0.0])
        y = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        assert jnp.allclose(bsc := _biot_savart_cutoff(x, y), 0.0), (
            f"Returned {bsc}, but expected zero"
        )
