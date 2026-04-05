from __future__ import annotations

from _operator import mul
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Sequence, overload

from jax import Array, numpy as jnp

from algebra.array_utils import ArrayListShape, ArrayList, check_arr_shape
from print_utils import warn, jax_print, VerbosityLevel
from utils import _make_pytree


@dataclass
class ConvergenceSettings:
    max_n_iter: Optional[int] = 40
    rel_disp_tol: Optional[float] = 1e-8
    abs_disp_tol: Optional[float] = 1e-12
    rel_force_tol: Optional[float] = 1e-8
    abs_force_tol: Optional[float] = 1e-12


@_make_pytree
class ConvergenceStatus:
    def __init__(self, convergence_settings: ConvergenceSettings):
        r"""
        Object to track convergence status of an iterative solver based on absolute and relative tolerances. Absolute
        convergence is measured for the delta vector, while relative convergence is measured as the ratio of the maximum
        element in the delta vector to the total vector.
        :param convergence_settings: Convergence settings object containing tolerances and maximum iteration count for
        convergence failure.
        """

        # make sure settings allow for loop to be broken
        if (
                convergence_settings.rel_disp_tol is None
                and convergence_settings.abs_disp_tol is None
                and convergence_settings.rel_force_tol is None
                and convergence_settings.abs_force_tol is None
        ):
            if convergence_settings.max_n_iter is None:
                raise ValueError(
                    "No convergence criteria provided, at least one tolerance or maximum iteration count "
                    "must be specified."
                )
            warn(
                "No convergence tolerances provided, will iterate until maximum iteration counter."
            )

        # base parameters
        self.i_iter: Array = jnp.zeros((), dtype=int)
        self.convergence_settings: ConvergenceSettings = convergence_settings

        # store residual values
        self.rel_disp_val: Array = jnp.zeros(())
        self.abs_disp_val: Array = jnp.zeros(())
        self.rel_force_val: Array = jnp.zeros(())
        self.abs_force_val: Array = jnp.zeros(())

        # convergence status
        self.converged: Array = jnp.zeros((), dtype=bool)
        self.converged_abs_disp: Array = jnp.zeros((), dtype=bool)
        self.converged_rel_disp: Array = jnp.zeros((), dtype=bool)
        self.converged_rel_force: Array = jnp.zeros((), dtype=bool)
        self.converged_abs_force: Array = jnp.zeros((), dtype=bool)

        # flags for other convergence failure modes
        self.final_iter: Array = jnp.zeros((), dtype=bool)
        self.has_nan: Array = jnp.zeros((), dtype=bool)

    def update(
            self,
            delta_disp: Optional[Array],
            total_disp: Optional[Array],
            delta_force: Optional[Array],
            total_force: Optional[Array],
    ) -> None:
        r"""
        :param delta_disp: Difference in displacement vector between current and previous iteration.
        :param total_disp: Total displacement for time step, used for relative convergence calculation, typically the
        current solution vector.
        :param delta_force: Residual force vector, used for force convergence calculation.
        :param total_force: Total force vector for time step, used for relative convergence calculation.
        """
        # update iteration counter
        self.i_iter += 1

        # check absolute displacement convergence
        if delta_disp is not None:
            self.abs_disp_val = self.abs_disp_val.at[...].set(jnp.linalg.norm(delta_disp))

            # NaNs are checked for with displacement magnitude
            self.has_nan = self.has_nan.at[...].set(jnp.isnan(delta_disp).any())

        if self.convergence_settings.abs_disp_tol is not None:
            self.converged_abs_disp = (
                    self.abs_disp_val < self.convergence_settings.abs_disp_tol
            )

        # check relative displacement convergence:
        if self.convergence_settings.rel_disp_tol is not None:
            if total_disp is None: raise ValueError("total_disp cannot be None")
            max_total_elem = jnp.linalg.norm(total_disp)
            self.rel_disp_val = self.rel_disp_val.at[...].set(
                self.abs_disp_val / max_total_elem
            )
            self.converged_rel_disp = self.converged_rel_disp.at[...].set(
                jnp.nan_to_num(self.rel_disp_val, True, jnp.inf)
                < self.convergence_settings.rel_disp_tol
            )

        # check absolute force convergence
        if self.convergence_settings.abs_force_tol is not None:
            if delta_force is None: raise ValueError("delta_force cannot be None")
            self.abs_force_val = self.abs_force_val.at[...].set(
                jnp.linalg.norm(delta_force)
            )
            self.converged_abs_force = (
                    self.abs_force_val < self.convergence_settings.abs_force_tol
            )

        # check relative force convergence:
        if self.convergence_settings.rel_force_tol is not None:
            if total_force is None: raise ValueError("total_force cannot be None")
            max_total_elem = jnp.abs(total_force).max()
            self.rel_force_val = self.rel_force_val.at[...].set(
                self.abs_force_val / max_total_elem
            )
            self.converged_rel_force = self.converged_rel_force.at[...].set(
                jnp.nan_to_num(self.rel_force_val, True, jnp.inf)
                < self.convergence_settings.rel_force_tol
            )

        # find convergence status of numerics (excluding failure modes such as max iterations or nans)
        self.converged = self.converged.at[...].set(
            (
                    self.converged_rel_disp
                    | self.converged_abs_disp
                    | self.converged_rel_force
                    | self.converged_abs_force
            )
            & (self.i_iter > 0)
        )

        # check for failure modes
        if self.convergence_settings.max_n_iter is not None:
            self.final_iter = self.final_iter.at[...].set(
                (self.i_iter >= self.convergence_settings.max_n_iter)
            )

    def get_status(self) -> Array:
        """Get overall convergence status."""
        return self.converged | self.has_nan | self.final_iter

    def reset_status(self) -> None:
        r"""
        Reset convergence status for next load step, setting all convergence flags to False.
        """
        self.converged = self.converged.at[...].set(False)
        self.converged_abs_disp = self.converged_abs_disp.at[...].set(False)
        self.converged_rel_disp = self.converged_rel_disp.at[...].set(False)
        self.converged_rel_force = self.converged_rel_force.at[...].set(False)
        self.converged_abs_force = self.converged_abs_force.at[...].set(False)
        self.rel_disp_val = self.rel_disp_val.at[...].set(0.0)
        self.abs_disp_val = self.abs_disp_val.at[...].set(0.0)
        self.rel_force_val = self.rel_force_val.at[...].set(0.0)
        self.abs_force_val = self.abs_force_val.at[...].set(0.0)
        self.has_nan = self.has_nan.at[...].set(False)
        self.final_iter = self.final_iter.at[...].set(False)
        self.i_iter = self.i_iter.at[...].set(0)

    def print_struct_message(self, t: Optional[Array], i_load_step: int) -> None:
        """Print convergence message for structure_dv based on status."""

        if t is None:
            # static message
            jax_print(
                "| Struct: {i_iter:<3} | {conv!s:<5} | {rel_disp_val:.03e} | {abs_disp_val:.03e} | "
                "{rel_force_val:.03e} | {abs_force_val:.03e} | {i_load_step:<2}        |",
                verbose_level=VerbosityLevel.NORMAL,
                i_load_step=i_load_step,
                i_iter=self.i_iter,
                conv=self.converged,
                rel_disp_val=self.rel_disp_val,
                abs_disp_val=self.abs_disp_val,
                rel_force_val=self.rel_force_val,
                abs_force_val=self.abs_force_val,
            )
        else:
            # dynamic message
            jax_print(
                "| {t:.03e} | Struct: {i_iter:<3} | {conv!s:<5} | {rel_disp_val:.03e} | {abs_disp_val:.03e} | "
                "{rel_force_val:.03e} | {abs_force_val:.03e} | {i_load_step:<2}        |",
                verbose_level=VerbosityLevel.NORMAL,
                t=t,
                i_load_step=i_load_step,
                i_iter=self.i_iter,
                conv=self.converged,
                rel_disp_val=self.rel_disp_val,
                abs_disp_val=self.abs_disp_val,
                rel_force_val=self.rel_force_val,
                abs_force_val=self.abs_force_val,
            )

    def print_fsi_message(self, t: Optional[Array]) -> None:
        """Print convergence message for structure_dv based on status."""

        if t is None:
            # static message
            jax_print(
                "| FSI: {i_iter:<3}    | {conv!s:<5} | {rel_disp_val:.03e} | {abs_disp_val:.03e} | {rel_force_val:.03e} | "
                "{abs_force_val:.03e} |           |",
                verbose_level=VerbosityLevel.NORMAL,
                i_iter=self.i_iter,
                conv=self.converged,
                rel_disp_val=self.rel_disp_val,
                abs_disp_val=self.abs_disp_val,
                rel_force_val=self.rel_force_val,
                abs_force_val=self.abs_force_val,
            )
        else:
            # dynamic message
            jax_print(
                "| {t:.03e} | FSI: {i_iter:<3}    | {conv!s:<5} | {rel_disp_val:.03e} | {abs_disp_val:.03e} | "
                "{rel_force_val:.03e} | {abs_force_val:.03e} |           |",
                verbose_level=VerbosityLevel.NORMAL,
                t=t,
                i_iter=self.i_iter,
                conv=self.converged,
                rel_disp_val=self.rel_disp_val,
                abs_disp_val=self.abs_disp_val,
                rel_force_val=self.rel_force_val,
                abs_force_val=self.abs_force_val,
            )

    def print_fsi_header(self, dynamic: bool) -> None:
        if dynamic:
            print("\n| Dynamic Solve                                                                               |")
            print("| Time      | Iter        | Conv  | Rel Disp  | Abs Disp  | Rel Force | Abs Force | Load Step |")
        else:
            print("\n| Static Solve                                                                    |")
            print("| Iter        | Conv  | Rel Disp  | Abs Disp  | Rel Force | Abs Force | Load Step |")

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ("convergence_settings",)

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "i_iter",
            "rel_disp_val",
            "abs_disp_val",
            "rel_force_val",
            "abs_force_val",
            "converged",
            "converged_abs_disp",
            "converged_rel_disp",
            "converged_rel_force",
            "converged_abs_force",
            "final_iter",
            "has_nan",
        )


class DesignVariables:
    def __init__(self):
        # should only be used in child classes
        self.shapes: dict[str, Optional[tuple[int, ...] | ArrayListShape]] = {}
        self.mapping: dict[str, Optional[Array | ArrayList]] = {}

    def get_vars(self) -> dict[str, Optional[Array | ArrayList]]:
        # should only be used in child classes
        return {}

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        # should only be used in child classes
        return ()

    def get_shapes(self) -> dict[str, Optional[tuple[int, ...] | ArrayListShape]]:
        def _elem_shape(
                elem: Optional[Array | ArrayList],
        ) -> Optional[tuple[int, ...] | ArrayListShape]:
            if elem is None:
                return None
            else:
                return elem.shape

        return {k: _elem_shape(var) for k, var in self.get_vars().items()}

    def make_index_mapping(self) -> tuple[dict[str, Optional[Array | ArrayList]], int]:
        mapping = {}
        cnt = 0
        for name, shape in self.shapes.items():
            if shape is not None:
                if isinstance(shape, tuple):
                    var_size = reduce(mul, shape, 1)
                    mapping[name] = jnp.arange(cnt, cnt + var_size).reshape(shape)
                    cnt += var_size
                elif isinstance(shape, ArrayListShape):
                    submappings = []
                    for i_arr in range(shape.n_arrays):
                        var_size = reduce(mul, shape.shapes[i_arr], 1)
                        submappings.append(
                            jnp.arange(cnt, cnt + var_size).reshape(shape.shapes[i_arr])
                        )
                        cnt += var_size
                    mapping[name] = ArrayList(submappings)
                else:
                    raise ValueError("Invalid shape type in DesignVariables.")
            else:
                mapping[name] = None
        return mapping, cnt

    def ravel_jacobian(self, f_size: int, x_size: int) -> Array:
        @overload
        def _inner_ravel(var: None) -> None:
            ...

        @overload
        def _inner_ravel(var: Array | ArrayList) -> Array:
            ...

        def _inner_ravel(var: Optional[Array | ArrayList]) -> Optional[Array]:
            if var is None:
                return None
            elif isinstance(var, Array):
                return var.reshape(f_size, -1)
            elif isinstance(var, ArrayList):
                return jnp.concatenate([_inner_ravel(subvar) for subvar in var])
            else:
                raise ValueError("Invalid variable type in DesignVariables.")

        arr = jnp.concatenate(
            [_inner_ravel(var) for var in self.get_vars().values() if var is not None],  # type: ignore
            axis=1,
        )
        check_arr_shape(arr, (f_size, x_size), "Internal jacobian")
        return arr

    def ravel(self) -> Array:
        return jnp.concatenate(
            [var.ravel() for var in self.get_vars().values() if var is not None]
        )

    def reshape(self, *args: int) -> Array:
        return self.ravel().reshape(*args)

    def from_adjoint(
            self, f_shape: tuple[int, ...], df_dx: Array
    ) -> dict[str, Array | ArrayList]:
        out_dict = {}
        for name in self.shapes.keys():
            if (this_mapping := self.mapping[name]) is not None:
                if isinstance((this_shapes := self.shapes[name]), tuple):
                    out_dict[name] = df_dx[:, this_mapping].reshape(
                        *f_shape, *this_shapes
                    )
                elif isinstance((this_shapes := self.shapes[name]), ArrayListShape):
                    subarrays = []
                    for i_arr in range(this_shapes.n_arrays):
                        subarrays.append(
                            df_dx[:, this_mapping[i_arr]].reshape(
                                *f_shape, *this_shapes.shapes[i_arr]
                            )
                        )
                    out_dict[name] = ArrayList(subarrays)
                else:
                    raise ValueError("Invalid shape type in DesignVariables.")
            else:
                out_dict[name] = None
        return out_dict

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "shapes", "mapping"
