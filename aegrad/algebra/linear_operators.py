from __future__ import annotations

from functools import singledispatchmethod
from typing import Optional, Callable, Sequence, TypeVar

from aegrad.aero.data_structures import InputUnflattened, StateUnflattened, OutputUnflattened
from aegrad.algebra.array_utils import check_arr_shape
from aegrad.print_output import print_with_time, warn

import jax
from jax import Array, numpy as jnp, jacobian


class LinearOperator:
    r"""
    Linear operator represented by a function, either as A(x) or Ax
    If the function is set to none, the zero matrix is assumed.
    """
    def __init__(
        self,
        func: Optional[Callable[[Array], Array]],
        shape: tuple[int, int],
        mat: Optional[Array] = None,
    ):

        if mat is not None:
            if mat.shape != shape:
                raise ValueError(f"Provided matrix has shape {mat.shape}, but expected shape {shape}.")

        self.func = func if func is not None else lambda x: jnp.zeros(shape[1])
        self.mat: Optional[Array] = mat
        self.shape = shape  # shapes of equivalent matrix

    def to_matrix(self) -> Array:
        if self.mat is not None:
            return self.mat
        else:
            # dL/dx at x at any point is the same, and should be independent of x
            self.mat = jacobian(lambda x_: self.func(x_), argnums=0)(jnp.full(self.shape[1], 1.0))
            return self.mat

    def __call__(self, rhs: "Array | LinearOperator") -> "Array | LinearOperator":
        if isinstance(rhs, LinearOperator):
            def new_func(x: Array) -> Array:
                return self.func(rhs.func(x))

            shape = (self.shape[0], rhs.shape[1])
            if self.mat is not None and rhs.mat is not None:
                new_mat = self.mat @ rhs.mat
            else:
                new_mat = None
            return LinearOperator(new_func, shape, new_mat)
        elif isinstance(rhs, Array):
            return self.func(rhs)
        else:
            raise TypeError("Incompatible type for multiplication with LinearOperator.")

    def __matmul__[T](self, rhs: T) -> T:
        return self(rhs)

    def __add__(self, rhs: "Array | LinearOperator") -> "Callable[[Array], Array] | LinearOperator":
        # runtime dispatch for addition as well
        if isinstance(rhs, LinearOperator):
            if self.shape != rhs.shape:
                raise ValueError("Cannot add LinearOperators with different shapes.")
            def new_func(x: Array) -> Array:
                return self.func(x) + rhs.func(x)
            if self.mat is not None and rhs.mat is not None:
                new_mat = self.mat + rhs.mat
            else:
                new_mat = None
            return LinearOperator(new_func, self.shape, new_mat)
        elif isinstance(rhs, Array):
            def new_func(x: Array) -> Array:
                return self.func(x) + rhs
            return new_func
        else:
            raise TypeError("Incompatible type for addition with LinearOperator.")



class BlockLinear:
    r"""
    Block linear operator represented by a function, supporting matmul
    """
    def __init__(self, entries: Sequence[Sequence[LinearOperator | Array]], mat: Optional[Array] = None):
        if not all(len(row) == len(entries[0]) for row in entries):
            raise ValueError("All rows in BlockLinear must have the same number of columns.")

        shapes: list[list[tuple[int, ...]]] = [[e.shape for e in row] for row in entries]    # [][][...]

        if any([any([len(e) != 2 for e in row]) for row in shapes]):
            raise ValueError("All entries in BlockLinear must be 2D arrays or linear operators.")

        shapes_arr = jnp.array(shapes, dtype=int)   # [n_block_row, n_block_col, 2]
        self.n_block_row: int = shapes_arr.shape[0]
        self.n_block_col: int = shapes_arr.shape[1]

        # every row must have equal column sizes and every column must have equal row sizes
        if not jnp.all(shapes_arr[..., 0] == shapes_arr[:, [0], 0]):
            raise ValueError("All columns in BlockLinear must have the same number of rows.")
        if not jnp.all(shapes_arr[..., 1] == shapes_arr[[0], :, 1]):
            raise ValueError("All rows in BlockLinear must have the same number of columns.")

        self.entries = entries

        # number of blocks in each dimension
        self.block_heights: Array = shapes_arr[:, 0, 0]   # [n_block_row]
        self.block_widths: Array = shapes_arr[0, :, 1]    # [n_block_col]
        self.shape: tuple[int, int] = (int(jnp.sum(self.block_heights)), int(jnp.sum(self.block_widths)))

        # index of entries for each block in the full matrix
        height_index = []
        i_start = 0
        for i_block_row in range(self.n_block_row):
            height_index.append(jnp.arange(i_start, i_start := i_start + self.block_heights[i_block_row]))
        self.height_index: tuple[Array, ...] = tuple(height_index)

        width_index = []
        i_start = 0
        for i_block_col in range(self.n_block_col):
            width_index.append(
                jnp.arange(
                    i_start, i_start := i_start + self.block_widths[i_block_col]
                )
            )
        self.width_index: tuple[Array, ...] = tuple(width_index)

        if mat is not None:
            if mat.shape != self.shape:
                raise ValueError(f"Provided matrix has shape {mat.shape}, but expected shape {self.shape}.")
        self.mat = mat

    def __matmul__(self, rhs: Array) -> Array:
        out = jnp.zeros(self.shape[0])
        for i_block_col in range(self.n_block_col):
            this_rhs = rhs[self.width_index[i_block_col]]
            for i_block_row in range(self.n_block_row):
                out = out.at[self.height_index[i_block_row]].add(self.entries[i_block_row][i_block_col] @ this_rhs)
        return out

    def get_matrix(self) -> Array:
        arrs = []
        for i_block_row in range(self.n_block_row):
            arrs.append([])
            for i_block_col in range(self.n_block_col):
                entry = self.entries[i_block_row][i_block_col]
                if isinstance(entry, LinearOperator):
                    arrs[-1].append(entry.to_matrix())
                else:
                    arrs[-1].append(entry)
        blk = jnp.block(arrs)
        self.mat = blk
        return blk


class LinearSystem:
    def __init__(self,
                 a: LinearOperator,
                 b: LinearOperator,
                 c: LinearOperator,
                 d: LinearOperator,
                 removed_u_np1: bool = False) -> None:
        self.a: LinearOperator = a
        self.b: LinearOperator = b
        self.c: LinearOperator = c
        self.d: LinearOperator = d
        self.n_inputs: int = b.shape[1]
        self.n_states: int = a.shape[0]
        self.n_outputs: int = c.shape[0]
        self.removed_u_np1: bool = removed_u_np1

    @print_with_time("Computing matrices for linear system...",
                     "Computed matrices for linear system in {:.2f} seconds.")
    def compute_matrices(self) -> None:
        self.a.to_matrix()
        self.b.to_matrix()
        self.c.to_matrix()
        self.d.to_matrix()

    @print_with_time("Removing u_np1 from linear system...",
                     "Removed u_np1 from linear system in {:.2f} seconds.")
    def remove_u_np1(self) -> None:
        if self.removed_u_np1:
            warn("u_np1 has already been removed from the system. Skipping.")
        else:
            self.d = (self.c @ self.b) + self.d
            self.b = self.a @ self.b
            self.removed_u_np1 = True

    @print_with_time(
        "Running linear system...",
        "Ran linear system in {:.2f} seconds.",
    )
    def run(self, u: Array, x0: Optional[Array] = None) -> tuple[Array, Array]:
        if not self.removed_u_np1:
            self.remove_u_np1()

        if x0 is not None:
            check_arr_shape(x0, (None, self.n_states), "x0")
        check_arr_shape(u, (None, self.n_inputs), "u")
        n_tstep = u.shape[0]

        def state_func(i_ts: int, x_: Array) -> Array:
            # jax.debug.print("Linear UVLM state step {i_ts}", i_ts=i_ts)
            return x_.at[i_ts, ...].set(self.a @ x_[i_ts - 1, ...] + self.b @ u[i_ts - 1, ...])

        x = jnp.zeros((n_tstep, self.n_states))
        if x0 is not None:
            x = x.at[0, ...].set(x0)
        x = jax.lax.fori_loop(1, n_tstep, state_func, x)

        def output_func(i_ts: int, y_: Array) -> Array:
            # jax.debug.print("Linear UVLM output step {i_ts}", i_ts=i_ts)
            return y_.at[i_ts, ...].set(self.c @ x[i_ts, ...] + self.d @ u[i_ts, ...])

        y = jnp.zeros((n_tstep, self.n_outputs))
        y = jax.lax.fori_loop(0, n_tstep, output_func, y)
        return x, y
