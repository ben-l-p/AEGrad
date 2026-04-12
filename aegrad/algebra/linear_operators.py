from __future__ import annotations
from typing import Optional, Callable, Sequence, overload

import jax
from jax import Array, numpy as jnp, jacobian

from algebra.array_utils import check_arr_shape
from print_utils import warn, jax_print
from utils import _make_pytree


@_make_pytree
class LinearOperator:
    r"""
    Linear operator represented by a function, either as A(x_target) or A @ x.
    """

    def __init__(
            self,
            func: Optional[Callable[[Array], Array]],
            shape: tuple[int, int],
            mat: Optional[Array] = None,
    ):
        r"""
        :param func: Function which represents the linear operator, [varphi] -> [m]
        :param shape: Shape of the equivalent matrix, (m, varphi)
        :param mat: If available, the explicit matrix representation of the operator
        """
        if mat is not None:
            if mat.shape != shape:
                raise ValueError(
                    f"Provided matrix has shape {mat.shape}, but expected shape {shape}."
                )

        self.func = func if func is not None else lambda x: jnp.zeros(shape[1])
        self._matrix: Optional[Array] = mat
        self.shape = shape  # arr_list_shapes of equivalent matrix

    @property
    def matrix(self) -> Array:
        r"""
        Obtain the matrix representation of the linear operator.
        :return: Matrix representation of the linear operator
        """
        if self._matrix is None:
            self.generate_matrix()
        return self._matrix  # type: ignore

    def generate_matrix(self) -> None:
        r"""
        Generate the matrix representation of the linear operator.
        """
        # dL/dx at x_target at any point is the same, and should be independent of x_target
        self._matrix = jacobian(lambda x_: self.func(x_), argnums=0)(
            jnp.full(self.shape[1], 1.0)
        )

    def __call__[T: Array | LinearOperator](self, rhs: T) -> T:
        r"""
        Evaluate the linear operator on an array or compose with another linear operator.
        :param rhs: Either an array to apply the operator to, or another linear operator to compose with.
        :return: The result of applying the operator to the array, or the composed linear operator.
        """
        if isinstance(rhs, LinearOperator):
            def new_func(x: Array) -> Array:
                return self.func(rhs.func(x))  # type: ignore

            shape = (self.shape[0], rhs.shape[1])
            if self._matrix is not None and rhs._matrix is not None:
                new_mat = self._matrix @ rhs._matrix
            else:
                new_mat = None
            return LinearOperator(new_func, shape, new_mat)
        elif isinstance(rhs, Array):
            return self.func(rhs)
        else:
            raise TypeError("Incompatible type for multiplication with LinearOperator.")

    def __matmul__[T: Array | LinearOperator](self, rhs: T) -> T:
        r"""
        Matrix multiplication operator overload, calls __call__ internally.
        :param rhs: Either an array to apply the operator to, or another linear operator to compose with.
        :return: The result of applying the operator to the array, or the composed linear operator.
        """
        return self(rhs)

    @overload
    def __add__(self, rhs: LinearOperator) -> LinearOperator:
        ...

    @overload
    def __add__(self, rhs: Array) -> Callable[[Array], Array]:
        ...

    def __add__(
            self, rhs: Array | LinearOperator
    ) -> Callable[[Array], Array] | LinearOperator:
        # runtime dispatch for addition as well
        if isinstance(rhs, LinearOperator):
            if self.shape != rhs.shape:
                raise ValueError(
                    "Cannot add LinearOperators with different arr_list_shapes."
                )

            def new_func(x: Array) -> Array:
                if isinstance(rhs, LinearOperator):
                    return self.func(x) + rhs.func(x)
                else:
                    raise ValueError("Incompatible type for addition with LinearOperator.")

            if self._matrix is not None and rhs._matrix is not None:
                new_mat = self._matrix + rhs._matrix
            else:
                new_mat = None
            return LinearOperator(new_func, self.shape, new_mat)
        elif isinstance(rhs, Array):

            def new_func(x: Array) -> Array:
                return self.func(x) + rhs

            return new_func
        else:
            raise TypeError("Incompatible type for addition with LinearOperator.")

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Return the names of all static methods for pytree serialization.
        :return: Sequence of static method names.
        """
        return ("shape",)

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Return the names of all dynamic methods for pytree serialization.
        :return: Sequence of dynamic method names.
        """
        return "func", "_matrix"


class BlockLinear:
    r"""
    Block linear matrix operator composed of smaller linear operators or arrays.
    """

    def __init__(
            self,
            entries: Sequence[Sequence[LinearOperator | Array]],
            mat: Optional[Array] = None,
    ):
        r"""
        Construct a BlockLinear operator from smaller linear operators or arrays.
        :param entries: Double nested sequence of linear operators or arrays representing the blocks.
        :param mat: Optional full matrix representation of the block linear operator.
        """
        if not all(len(row) == len(entries[0]) for row in entries):
            raise ValueError(
                "All rows in BlockLinear must have the same number of columns."
            )

        shapes: list[list[tuple[int, ...]]] = [
            [e.shape for e in row] for row in entries
        ]  # [][][...]

        if any([any([len(e) != 2 for e in row]) for row in shapes]):
            raise ValueError(
                "All entries in BlockLinear must be 2D arrays or linear operators."
            )

        shapes_arr = jnp.array(shapes, dtype=int)  # [n_block_row, n_block_col, 2]
        self.n_block_row: int = shapes_arr.shape[0]
        self.n_block_col: int = shapes_arr.shape[1]

        # every row must have equal column sizes and every column must have equal row sizes
        if not jnp.all(shapes_arr[..., 0] == shapes_arr[:, [0], 0]):
            raise ValueError(
                "All columns in BlockLinear must have the same number of rows."
            )
        if not jnp.all(shapes_arr[..., 1] == shapes_arr[[0], :, 1]):
            raise ValueError(
                "All rows in BlockLinear must have the same number of columns."
            )

        self.entries = entries

        # number of blocks in each dimension
        self.block_heights: Array = shapes_arr[:, 0, 0]  # [n_block_row]
        self.block_widths: Array = shapes_arr[0, :, 1]  # [n_block_col]
        self.shape: tuple[int, int] = (
            int(jnp.sum(self.block_heights)),
            int(jnp.sum(self.block_widths)),
        )

        # index of entries for each block in the full matrix
        height_index = []
        i_start = 0
        for i_block_row in range(self.n_block_row):
            height_index.append(
                jnp.arange(
                    i_start, i_start := i_start + self.block_heights[i_block_row]
                )
            )
        self.height_index: tuple[Array, ...] = tuple(height_index)

        width_index = []
        i_start = 0
        for i_block_col in range(self.n_block_col):
            width_index.append(
                jnp.arange(i_start, i_start := i_start + self.block_widths[i_block_col])
            )
        self.width_index: tuple[Array, ...] = tuple(width_index)

        if mat is not None:
            if mat.shape != self.shape:
                raise ValueError(
                    f"Provided matrix has shape {mat.shape}, but expected shape {self.shape}."
                )
        self.mat = mat

    def __matmul__(self, rhs: Array) -> Array:
        r"""
        Matrix multiplication operator overload for BlockLinear operator.
        :param rhs: Right-hand side array to multiply with.
        :return: Resulting array after multiplication.
        """
        out = jnp.zeros(self.shape[0])
        for i_block_col in range(self.n_block_col):
            this_rhs = rhs[self.width_index[i_block_col]]
            for i_block_row in range(self.n_block_row):
                out = out.at[self.height_index[i_block_row]].add(
                    self.entries[i_block_row][i_block_col] @ this_rhs
                )
        return out

    def get_matrix(self) -> Array:
        r"""
        Generate and return the full matrix representation of the BlockLinear operator.
        :return: Full matrix representation of the BlockLinear operator.
        """
        arrs = []
        for i_block_row in range(self.n_block_row):
            arrs.append([])
            for i_block_col in range(self.n_block_col):
                entry = self.entries[i_block_row][i_block_col]
                if isinstance(entry, LinearOperator):
                    arrs[-1].append(entry.matrix)
                else:
                    arrs[-1].append(entry)
        blk = jnp.block(arrs)
        self.mat = blk
        return blk


@_make_pytree
class LinearSystem:
    r"""
    Linear system represented in state-space form, with tools for time-stepping
    """

    def __init__(
            self,
            a: LinearOperator,
            b: LinearOperator,
            c: LinearOperator,
            d: LinearOperator,
            removed_u_np1: bool = False,
    ) -> None:
        r"""
        Initialise the LinearSystem with state-space linear operators.
        :param a: System matrix A
        :param b: Input matrix B
        :param c: Output matrix C
        :param d: Feedthrough matrix D
        :param removed_u_np1: If true, indicates that the system is in terms of inputs at time varphi only.
        """
        self.a: LinearOperator = a
        self.b: LinearOperator = b
        self.c: LinearOperator = c
        self.d: LinearOperator = d
        self.n_inputs: int = b.shape[1]
        self.n_states: int = a.shape[0]
        self.n_outputs: int = c.shape[0]
        self.removed_u_np1: bool = removed_u_np1

    def generate_matrices(self) -> None:
        r"""
        Compute the matrix representations of the linear operators in the system.
        """
        self.a.generate_matrix()
        self.b.generate_matrix()
        self.c.generate_matrix()
        self.d.generate_matrix()

    def remove_u_np1(self) -> None:
        r"""
        Remove the dependence on u at time varphi+1 from the linear system, modifying b and d accordingly.
        :math:`D_{new} = C B + D` and :math:`B_{new} = A B`
        """
        if self.removed_u_np1:
            warn("u_np1 has already been removed from the system. Skipping.")
        else:
            self.d = (self.c @ self.b) + self.d
            self.b = self.a @ self.b
            self.removed_u_np1 = True

    def run(
            self, u: Array, x0: Optional[Array] = None, use_matrix=False
    ) -> tuple[Array, Array]:
        r"""
        Run the linear system for a time history of input vector u.
        :param u: Time history of input vectors, shape [n_tstep, n_inputs]
        :param x0: Initial state vector, [n_states]. If None, assumed to be zero.
        :param use_matrix: If true, use the matrix representations of the linear operators.
        :return: State history and output history, arr_list_shapes [n_tstep, n_states] and [n_tstep, n_outputs]
        """
        if x0 is not None:
            check_arr_shape(x0, (None, self.n_states), "x0")
        check_arr_shape(u, (None, self.n_inputs), "u")
        n_tstep = u.shape[0]

        if use_matrix:
            a, b, c, d = self.a.matrix, self.b.matrix, self.c.matrix, self.d.matrix
        else:
            a, b, c, d = self.a, self.b, self.c, self.d

        def state_func(i_ts: int, x_: Array) -> Array:
            r"""
            State update function for time step i_ts, given as :math:`x_{varphi} = A x_{varphi-1} + B u_{varphi-1}` or :math:`x_n = A x_{varphi-1} + B u_n`.
            :param i_ts: Time step index to obtain new states for.
            :param x_: State history array being updated, [n_tstep, n_states]
            :return: Updated state history array, [n_tstep, n_states]
            """
            jax_print("Linear UVLM state step {i_ts}", i_ts=i_ts)
            this_u = u[i_ts - 1, ...] if self.removed_u_np1 else u[i_ts, ...]
            return x_.at[i_ts, ...].set(a @ x_[i_ts - 1, ...] + b @ this_u)

        x = jnp.zeros((n_tstep, self.n_states))
        if x0 is not None:
            x = x.at[0, ...].set(x0)
        x = jax.lax.fori_loop(1, n_tstep, state_func, x)

        def output_func(i_ts: int, y_: Array) -> Array:
            r"""
            Output computation function for time step i_ts, given as :math:`y_n = C x_n + D u_n`.
            :param i_ts: Time step index to obtain outputs for.
            :param y_: Output history array being updated, [n_tstep, n_outputs]
            :return: Updated output history array, [n_tstep, n_outputs]
            """
            jax_print("Linear UVLM output step {i_ts}", i_ts=i_ts)
            return y_.at[i_ts, ...].set(c @ x[i_ts, ...] + d @ u[i_ts, ...])

        y = jnp.zeros((n_tstep, self.n_outputs))
        y = jax.lax.fori_loop(0, n_tstep, output_func, y)
        return x, y

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Output the names of all static attributes for pytree serialization.
        :return: Sequence of static attribute names.
        """
        return "n_inputs", "n_states", "n_outputs"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Output the names of all dynamic attributes for pytree serialization.
        :return: Sequence of dynamic attribute names.
        """
        return "a", "b", "c", "d", "removed_u_np1"
