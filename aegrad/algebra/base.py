from jax import numpy as jnp
from jax import Array, jacobian
from jax.lax import cond
from typing import Callable, Sequence, Optional


def matrix2(mat: Array) -> Array:
    return mat @ mat

def clip_to_pi(val: Array):
    # [] -> []
    # clips a value to be within [-pi, pi]
    return jnp.arctan2(jnp.sin(val), jnp.cos(val))

def chi(rmat: Array) -> Array:
    # [3, 3] -> [6, 6]
    return jnp.block([[rmat, jnp.zeros((3, 3))], [jnp.zeros((3, 3)), rmat]])

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

        self.func = func if func is None else lambda x: jnp.zeros(shape[1])
        self.mat: Optional[Array] = mat if func is not None else jnp.zeros(shape)
        self.shape = shape  # shape of equivelent matrix

    def get_matrix(self) -> Array:
        if self.mat is not None:
            return self.mat
        else:
            # the nan values should go away with it being linear
            # dL/dx at x at any point is the same, and should be independent of x
            return jacobian(self.func)(jnp.full(self.shape[1], jnp.nan))

    def __matmul__(self, rhs: Array) -> Array:
        return self.func(rhs)

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
                    arrs[-1].append(entry.get_matrix())
                else:
                    arrs[-1].append(entry)
        blk = jnp.block(arrs)
        self.mat = blk
        return blk

def finite_difference(
    i_: int, data: Array, delta: Array, axis: int, order: int = 1
) -> Array:
    r"""
    Compute the finite difference of the data at a given time step. This assumes that data[:i_+1] is available.
    :param i_: Index of derivative to obtain.
    :param data: Data to compute the finite difference on, [...].
    :param delta: Small perturbation value for finite difference, which divides the difference.
    :param axis: Axis along which to compute the finite difference.
    :param order: Order of the finite difference (1 or 2).
    :return: Finite difference of the data at the specified time step.
    """

    if order not in (0, 1, 2):
        raise ValueError("Order must be 0, 1, or 2.")

    def _slice_order(shift_: int) -> tuple[slice | int, ...]:
        sl: list[slice | int] = [slice(None)] * data.ndim
        sl[axis] = i_ - shift_
        return tuple(sl)

    def _order0() -> Array:
        return jnp.zeros([n for i, n in enumerate(data.shape) if i != axis])

    def _order1() -> Array:
        return (data[_slice_order(0)] - data[_slice_order(1)]) / delta

    def _order2() -> Array:
        return (
            3.0 * data[_slice_order(0)]
            - 4.0 * data[_slice_order(1)]
            + data[_slice_order(2)]
        ) / (2.0 * delta)

    def _err() -> Array:
        return jnp.full([n for i, n in enumerate(data.shape) if i != axis], jnp.nan)

    # use lower order when not enough data is available
    # for the instance where only a single data point is available, gradient is set to zero
    order = jnp.minimum(order, i_)
    return cond(
        order == 0,
        _order0,
        lambda: cond(
            order == 1, _order1, lambda: cond(order == 2, _order2, _err)
        ),
    )