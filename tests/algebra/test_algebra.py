from jax import numpy as jnp
from jax import Array

from aegrad.algebra.linear_operators import LinearOperator, BlockLinear
from aegrad.algebra.base import taylor_series


class TestLinearOperator:
    @staticmethod
    def test_identity_lin():
        r"""
        Test using a linear operator
        """
        vec = jnp.linspace(2.0, 6.0, 5)
        linop = LinearOperator(lambda x: x, shape=(5, 5))
        assert jnp.allclose(out := linop @ vec, vec), f"Expected {vec}, but got {out}"

    @staticmethod
    def test_obtain_mat_jac():
        r"""
        Test obtaining the linear matrix from autograd
        """
        linop = LinearOperator(lambda x: x, shape=(5, 5))
        mat = linop.matrix
        assert jnp.allclose(mat, jnp.eye(5)), f"Expected identity matrix, but got {mat}"

    @staticmethod
    def test_obtain_mat_input():
        r"""
        Test obtaining the linear matrix from input_
        """
        linop = LinearOperator(lambda x: x, shape=(5, 5), mat=jnp.eye(5))
        mat = linop.matrix
        assert jnp.allclose(mat, jnp.eye(5)), f"Expected identity matrix, but got {mat}"


class TestBlockLinear:
    @staticmethod
    def test_incorrect_block_shapes():
        try:
            BlockLinear([[jnp.eye(2), jnp.eye(3)], [jnp.eye(2), jnp.eye(2)]])
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError due to incorrect block arr_list_shapes, but none was raised."
            )

    @staticmethod
    def test_internal_shapes():
        input_ = [
            [jnp.zeros((2, 1)), jnp.zeros((2, 4))],
            [jnp.zeros((5, 1)), jnp.zeros((5, 4))],
        ]

        blk = BlockLinear(input_)

        assert blk.n_block_row == 2, f"Expected 2 block rows, but got {blk.n_block_row}"
        assert blk.n_block_col == 2, (
            f"Expected 2 block columns, but got {blk.n_block_col}"
        )

        assert jnp.allclose(blk.block_heights, jnp.array([2, 5])), (
            f"Expected block heights [2, 5], but got {blk.block_heights}"
        )
        assert jnp.allclose(blk.block_widths, jnp.array([1, 4])), (
            f"Expected block widths [1, 4], but got {blk.block_widths}"
        )

        exp = (jnp.arange(2), jnp.arange(2, 7))
        assert all([jnp.allclose(i, j) for i, j in zip(blk.height_index, exp)]), (
            f"Unexpected height index, expected {exp}, returned {blk.height_index}"
        )

        exp = (jnp.arange(1), jnp.arange(1, 5))
        assert all([jnp.allclose(i, j) for i, j in zip(blk.width_index, exp)]), (
            f"Unexpected height index, expected {exp}, returned {blk.width_index}"
        )

        assert blk.shape == (7, 5), f"Expected shape (7, 5), but got {blk.shape}"

    @staticmethod
    def test_matmul():
        input_ = [[jnp.eye(3), jnp.zeros((3, 5))], [jnp.zeros((5, 3)), jnp.eye(5)]]

        blk = BlockLinear(input_)
        vec = jnp.linspace(2.0, 6.0, 8)
        assert jnp.allclose(out := blk @ vec, vec), f"Expected {vec}, but got {out}"
        assert jnp.allclose(blk.get_matrix(), jnp.eye(8)), (
            f"Expected identity matrix, but got {blk.get_matrix()}"
        )

    @staticmethod
    def test_linear():
        l1 = LinearOperator(lambda x: x, shape=(3, 3))
        l2 = LinearOperator(lambda x: jnp.zeros(3), shape=(3, 5))
        l3 = LinearOperator(lambda x: jnp.zeros(5), shape=(5, 3))
        l4 = LinearOperator(lambda x: x, shape=(5, 5))

        input_ = [[l1, l2], [l3, l4]]

        blk = BlockLinear(input_)
        vec = jnp.linspace(2.0, 6.0, 8)
        assert jnp.allclose(out := blk @ vec, vec), f"Expected {vec}, but got {out}"
        assert jnp.allclose(blk.get_matrix(), jnp.eye(8)), (
            f"Expected identity matrix, but got {blk.get_matrix()}"
        )

    @staticmethod
    def test_mixed():
        l1 = LinearOperator(lambda x: x, shape=(3, 3))
        l4 = LinearOperator(lambda x: x, shape=(5, 5))

        input_ = [[l1, jnp.zeros((3, 5))], [jnp.zeros((5, 3)), l4]]

        blk = BlockLinear(input_)
        vec = jnp.linspace(2.0, 6.0, 8)
        assert jnp.allclose(out := blk @ vec, vec), f"Expected {vec}, but got {out}"
        assert jnp.allclose(blk.get_matrix(), jnp.eye(8)), (
            f"Expected identity matrix, but got {blk.get_matrix()}"
        )

    @staticmethod
    def test_single():
        blk = BlockLinear([[jnp.eye(4)]])
        vec = jnp.linspace(2.0, 6.0, 4)
        assert jnp.allclose(out := blk @ vec, vec), f"Expected {vec}, but got {out}"

    @staticmethod
    def test_rectangular():
        full_arr = jnp.arange(24, dtype=float).reshape((4, 6))
        arr1 = full_arr[:1, :4]  # [1, 4]
        arr2 = full_arr[:1, 4:]  # [1, 2]
        arr3 = full_arr[1:, :4]  # [3, 4]
        arr4 = full_arr[1:, 4:]  # [3, 2]

        blk = BlockLinear([[arr1, arr2], [arr3, arr4]])
        vec = jnp.linspace(8.0, 9.0, 6)

        exp = full_arr @ vec
        assert jnp.allclose(out := blk @ vec, exp), f"Expected {exp}, but got {out}"


class TestTaylorSeries:
    @staticmethod
    def test_identity():
        r"""
        Check that the Taylor series expansion of a function evaluated at the expansion point is equal to the function
        value at that point"""

        def func(x_: Array) -> Array:
            return x_

        x = jnp.array([1.0, 2.0, 3.0])

        taylor_x = taylor_series(func, x, 5)(x)

        assert jnp.allclose(x, taylor_x), f"Expected {x}, but got {taylor_x}"

    @staticmethod
    def test_linear():
        r"""
        Check that the Taylor series expansion of a linear function is equal to the function itself, regardless of
        the expansion point.
        """

        def func(x_: Array) -> Array:
            return 2.0 * x_

        x0 = jnp.zeros((1,))

        x = jnp.array([4.0])
        f_x = func(x)

        taylor_x = taylor_series(func, x0, 1)(x)

        assert jnp.allclose(f_x, taylor_x), f"Expected {x}, but got {taylor_x}"

    @staticmethod
    def test_quadratic():
        r"""
        Check that the Taylor series expansion of a quadratic function is equal to the function itself, regardless of
        the expansion point, when the order of the Taylor series expansion is 2 or higher.
        """

        def func(x_: Array) -> Array:
            return 3.0 * x_**2 + 8.0 * x_ + 7.0

        x0 = jnp.array([3.0, 5.0])

        x = jnp.array([4.0, 7.0])
        f_x = func(x)

        taylor_x = taylor_series(func, x0, 10)(x)

        assert jnp.allclose(f_x, taylor_x), f"Expected {x}, but got {taylor_x}"
