from __future__ import annotations

from typing import Any, Sequence, Optional
from typing import Protocol, TypeVar
from dataclasses import fields, is_dataclass, dataclass
from jax import tree_util, Array, numpy as jnp

from print_output import warn, jax_print, VerbosityLevel


class SupportsPytree(Protocol):
    def _dynamic_names(self) -> Sequence[str]: ...
    def _static_names(self) -> Sequence[str]: ...


T = TypeVar("T", bound=SupportsPytree)


def _make_pytree(cls: type[T]) -> type[T]:
    """
    Convert an object to a pytree structure.
    :param cls: Class to be converted to a pytree.
    """

    def flatten_func(self: T) -> tuple[tuple[Any], tuple[Any]]:
        children = tuple(getattr(self, field) for field in self._dynamic_names())
        aux_data = tuple(getattr(self, field) for field in self._static_names())
        return children, aux_data

    def unflatten_func(aux_data: tuple[Any], children: tuple[Any]) -> T:
        obj = cls.__new__(cls)  # Create an uninitialized instance
        for field_name, value in zip(cls._dynamic_names(), children):
            setattr(obj, field_name, value)
        for field_name, value in zip(cls._static_names(), aux_data):
            setattr(obj, field_name, value)
        return obj

    tree_util.register_pytree_node(cls, flatten_func, unflatten_func)
    return cls


def _check_type(obj: Any, type_: type | Sequence[type]) -> None:
    try:
        len(type_)
    except TypeError:
        type_ = (type_,)

    for t in type_:
        if isinstance(obj, t):
            return

    raise TypeError(f"Expected {type_}, but got {obj}.")


def _shallow_asdict(obj):
    if not is_dataclass(obj):
        raise TypeError("object must be a dataclass")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


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
        self.abs_disp_val = self.abs_disp_val.at[...].set(jnp.linalg.norm(delta_disp))
        if self.convergence_settings.abs_disp_tol is not None:
            self.converged_abs_disp = (
                self.abs_disp_val < self.convergence_settings.abs_disp_tol
            )

        # check relative displacement convergence:
        if self.convergence_settings.rel_disp_tol is not None:
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
            self.abs_force_val = self.abs_force_val.at[...].set(
                jnp.linalg.norm(delta_force)
            )
            self.converged_abs_force = (
                self.abs_force_val < self.convergence_settings.abs_force_tol
            )

        # check relative force convergence:
        if self.convergence_settings.rel_force_tol is not None:
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
        self.has_nan = self.has_nan.at[...].set(jnp.isnan(delta_disp).any())

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
        """Print convergence message for structure based on status."""

        if t is None:
            # static message
            jax_print(
                "| Load Step: {i_load_step:<2} | Iter: {i_iter:<3} | Conv: {conv:1} | Rel. Disp: "
                "{rel_disp_val:.02e} | Abs. Disp: {abs_disp_val:.02e} | Rel. Force: {rel_force_val:.02e} | Abs. Force: "
                "{abs_force_val:.02e} |",
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
                "| Time: {t:.03e} | Load Step: {i_load_step:<2} | Iter: {i_iter:<3} | Conv: {conv:1} | Rel. Disp.: {rel_disp_val:.02e} | Abs. "
                "Disp: {abs_disp_val:.02e} | Rel. Force: {rel_force_val:.02e} | Abs. Force: {abs_force_val:.02e} |",
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
        """Print convergence message for structure based on status."""

        if t is None:
            # static message
            jax_print(
                "| FSI Iter: {i_iter:<3}             | Conv: {conv:1} | Rel. Disp: "
                "{rel_disp_val:.02e} | Abs. Disp: {abs_disp_val:.02e} | Rel. Force: {rel_force_val:.02e} | Abs. Force: "
                "{abs_force_val:.02e} |",
                verbose_level=VerbosityLevel.NORMAL,
                i_iter=self.i_iter,
                conv=self.converged,
                rel_disp_val=self.rel_disp_val,
                abs_disp_val=self.abs_disp_val,
                rel_force_val=self.rel_force_val,
                abs_force_val=self.abs_force_val,
            )
        else:
            raise NotImplementedError

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
