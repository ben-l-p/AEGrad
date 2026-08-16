from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from math import factorial
from typing import TYPE_CHECKING, Any, Literal

import jax
from jax import Array, jacfwd, jacrev
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.lax import cond
from jax.scipy.special import bernoulli

from flapjax.utils.constants import BASE_SUMMATION_ORDER
from flapjax.utils.utils import conditional_profile

ADMode = Literal["reverse", "forward"]

if TYPE_CHECKING:
    from flapjax.aero.gradients.data_structures import AeroJacobianApproximations
    from flapjax.structure.gradients.data_structures import BeamJacobianApproximations


def matrix2(mat: Array) -> Array:
    r"""
    Computes the square of a matrix.
    :param mat: Matrix, ``(n, n)``.
    :return: Matrix squared, ``(n, n)``.
    """
    return mat @ mat


def clip_to_pi(val: float | Array):
    r"""
    Clips an angle value to be within `[-pi, pi]`.
    :param val: Scalar to bound.
    :return: Bounded scalar within `[-pi, pi]`.
    """
    return jnp.arctan2(jnp.sin(val), jnp.cos(val))


def chi(rmat: Array) -> Array:
    r"""
    Converts a 3x3 rotation matrix to a 6x6 matrix used in spatial transformations.
    :param rmat: Rotation matrix, ``(a, b)``.
    :return: Block matrix with diagonal rotation matrices, ``(2a, 2b)``.
    """
    return jnp.block([[rmat, jnp.zeros_like(rmat)], [jnp.zeros_like(rmat), rmat]])


def finite_difference(
    i_: int, data: Array, delta: Array, axis: int, order: int = 1
) -> Array:
    r"""
    Compute the finite difference of the data at a given time step. This assumes that ``data[:i_+1]`` is available.
    :param i_: Index of derivative to obtain.
    :param data: Data to compute the finite difference on, (...).
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

    # use lower int_order when not enough data is available
    # for the instance where only a single data point is available, gradient is set to zero
    order: Array = jnp.array((order, i_)).min()
    return cond(
        order == 0,
        _order0,
        lambda: cond(order == 1, _order1, lambda: cond(order == 2, _order2, _err)),
    )


def exp_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the matrix exponential using truncated summation.
    :param a: Algebra matrix to exponentiate, ``(n, n)``.
    :param order: Order of summation.
    :return: Exponential of matrix, ``(n, n)``
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += jnp.linalg.matrix_power(a, i) / factorial(i)
    return result


def log_sum(g: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the matrix logarithm using truncated summation.
    :param g: Group matrix to exponentiate, ``(n, n)``.
    :param order: Order of summation.
    :return: Logarithm of matrix, ``(n, n)``
    """

    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("Input must be a square matrix")

    g_e = g - jnp.eye(g.shape[0])
    result = g_e

    for i in range(2, order + 1):
        result += (-1.0) ** (i + 1) * jnp.linalg.matrix_power(g_e, i) / i
    return result


def t_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the tangent operator truncated summation. This is used to validate other implementations.
    :param a: Adjoint action matrix, ``(n, n)``
    :param order: Order of summation.
    :return: Tangent operator, ``(n, n)``
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += (-1.0) ** i * jnp.linalg.matrix_power(a, i) / factorial(i + 1)
    return result


def t_inv_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the inverse tangent operator truncated summation.
    :param a: Adjoint action matrix, ``(n, n)``
    :param order: Order of summation.
    :return: Inverse tangent operator, ``(n, n)``
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    b = bernoulli(order)

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += (-1.0) ** i * b[i] * jnp.linalg.matrix_power(a, i) / factorial(i)
    return result


def jacrev_kwargs(
    func: Callable[..., Array],
    argnames: str | Sequence[str],
    allow_int: bool = True,
) -> Callable[..., dict[str, Any]]:
    r"""
    Custom reverse Jacobian routine which allows for keyword arguments.
    :param func: Function for which to obtain Jacobians. Must be callable with the keyword arguments later passed to
    the returned function.
    :param argnames: Argument names of variables for which to obtain Jacobians. These must be a subset of the keyword
    argument names provided to the returned function.
    :param allow_int: As `jax.jacrev`.
    :return: Function that accepts keyword arguments and returns a dictionary of argname: Jacobian pairs.
    """
    return _jac_kwargs(func, argnames=argnames, mode="reverse", allow_int=allow_int)


def jacfwd_kwargs(
    func: Callable[..., Array],
    argnames: str | Sequence[str],
) -> Callable[..., dict[str, Any]]:
    r"""
    Forward-mode counterpart of :func:`jacrev_kwargs`. Prefer this when the input dimension for the selected
    arguments is smaller than the output dimension of ``func``, which happens for design-variable Jacobians and for
    dynamic-adjoint residual blocks whose input state size is smaller than the residual dimension.
    :param func: Function for which to obtain Jacobians.
    :param argnames: Argument names of variables for which to obtain Jacobians.
    :return: Function that accepts keyword arguments and returns a dictionary of argname: Jacobian pairs.
    """
    return _jac_kwargs(func, argnames=argnames, mode="forward", allow_int=False)


def _jac_kwargs(
    func: Callable[..., Array],
    argnames: str | Sequence[str],
    mode: ADMode,
    allow_int: bool,
) -> Callable[..., dict[str, Any]]:
    argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)

    def inner_func(**kwargs: Any) -> dict[str, Array]:
        full_argnames = list(kwargs.keys())
        argvals = list(kwargs.values())
        argnums = tuple(full_argnames.index(name) for name in argnames)

        def positional_adapter(*args: Any) -> Array:
            return func(**dict(zip(full_argnames, args)))

        if mode == "reverse":
            out_jacs = jacrev(positional_adapter, argnums=argnums, allow_int=allow_int)(
                *argvals
            )
        else:
            out_jacs = _jacfwd_allow_int(
                positional_adapter, argnums=argnums, args=argvals
            )
        return dict(zip(argnames, out_jacs))

    return inner_func


def _jacrev_via_map_kwargs(
    func: Callable[..., Array],
    argnames: str | Sequence[str],
    map_batch_size: int | None,
) -> Callable[..., dict[str, Any]]:
    r"""
    Reverse-mode Jacobian assembly that uses map to reduce memory usage compared to jax.jacrev.
    """
    argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)

    def inner_func(**kwargs: Any) -> dict[str, Array]:
        full_argnames = list(kwargs.keys())
        argvals = list(kwargs.values())
        argnums = tuple(full_argnames.index(name) for name in argnames)
        dyn_vals = tuple(argvals[i] for i in argnums)

        def positional_adapter(*dyn_args_: Any) -> Array:
            merged = list(argvals)
            for i, val in zip(argnums, dyn_args_):
                merged[i] = val
            return func(**dict(zip(full_argnames, merged)))

        y, pullback = jax.vjp(positional_adapter, *dyn_vals)
        assert isinstance(y, jax.Array), (
            "_jacrev_via_map_kwargs currently only supports Array-valued outputs"
        )
        m = int(y.size)
        basis = jnp.eye(m, dtype=y.dtype).reshape((m,) + y.shape)

        # account for case where these are zero-size arguments as these break map
        keep_idxs = tuple(
            i
            for i, v in enumerate(dyn_vals)
            if sum(int(leaf.size) for leaf in jax.tree.leaves(v)) > 0
        )

        def filtered_pullback(cotan: Any) -> tuple[Any, ...]:
            return tuple(pullback(cotan)[i] for i in keep_idxs)

        kept = (
            jax.lax.map(filtered_pullback, basis, batch_size=map_batch_size)
            if keep_idxs
            else ()
        )

        kept_iter = iter(kept)
        cotangents = tuple(
            next(kept_iter)
            if i in keep_idxs
            else jax.tree.map(
                lambda leaf: jnp.zeros(
                    (m,) + jnp.shape(leaf), dtype=jnp.result_type(leaf)
                ),
                v,
            )
            for i, v in enumerate(dyn_vals)
        )
        return dict(zip(argnames, cotangents))

    return inner_func


def _jacfwd_via_map_kwargs(
    func: Callable[..., Array],
    argnames: str | Sequence[str],
    map_batch_size: int | None,
) -> Callable[..., dict[str, Any]]:
    r"""
    Forward-mode Jacobian assembly that uses map to reduce memory usage compared to jax.jacfwd by allowing for an
    input batch size.
    """
    argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)

    def inner_func(**kwargs: Any) -> dict[str, Any]:
        full_argnames = list(kwargs.keys())
        argvals = list(kwargs.values())

        jacs: dict[str, Any] = {}
        for name in argnames:
            arg_idx = full_argnames.index(name)
            v = argvals[arg_idx]
            leaves, treedef = jax.tree.flatten(v)
            leaf_sizes = [int(leaf.size) for leaf in leaves]
            v_size = sum(leaf_sizes)

            def single_arg_adapter(dyn_arg: Any, _arg_idx: int = arg_idx) -> Array:
                merged = list(argvals)
                merged[_arg_idx] = dyn_arg
                return func(**dict(zip(full_argnames, merged)))

            if v_size == 0:
                y_struct = jax.eval_shape(single_arg_adapter, v)
                jacs[name] = jax.tree.unflatten(
                    treedef,
                    [
                        jnp.zeros(y_struct.shape + leaf.shape, dtype=leaf.dtype)
                        for leaf in leaves
                    ],
                )
                continue

            flat_v, unravel = ravel_pytree(v)
            flat_basis = jnp.eye(v_size, dtype=flat_v.dtype)

            def single_jvp(
                flat_tangent: Array,
                _adapter: Callable[..., Array] = single_arg_adapter,
                _primal: Any = v,
                _unravel: Callable[[Array], Any] = unravel,
            ) -> Array:
                _, out_tan = jax.jvp(_adapter, (_primal,), (_unravel(flat_tangent),))
                return out_tan

            stacked = jax.lax.map(single_jvp, flat_basis, batch_size=map_batch_size)
            y_shape = stacked.shape[1:]
            n_y = len(y_shape)

            split_points: list[int] = []
            running = 0
            for s in leaf_sizes[:-1]:
                running += s
                split_points.append(running)
            pieces = (
                jnp.split(stacked, split_points, axis=0) if split_points else [stacked]
            )

            leaf_jacs = []
            for piece, leaf in zip(pieces, leaves):
                reshaped = piece.reshape(leaf.shape + y_shape)
                n_leaf = len(leaf.shape)
                perm = tuple(range(n_leaf, n_leaf + n_y)) + tuple(range(n_leaf))
                leaf_jacs.append(jnp.transpose(reshaped, perm))

            jacs[name] = jax.tree.unflatten(treedef, leaf_jacs)

        return jacs

    return inner_func


def _jacfwd_allow_int(
    func: Callable[..., Array], argnums: Sequence[int], args: Sequence[Any]
) -> tuple[Any, ...]:
    r"""
    Forward-mode Jacobian that tolerates integer leaves in the argnums-selected pytrees, matching the semantics of
    :func:`jax.jacrev` with ``allow_int=True`` (integer leaves receive zero Jacobian columns).
    """
    # Splits: for each argnum, (leaves, treedef, float_mask)
    splits = []
    for arg_i in argnums:
        leaves, treedef = jax.tree_util.tree_flatten(args[arg_i])
        float_mask = [jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves]
        splits.append((leaves, treedef, float_mask))

    def call_from_float_leaves(*float_leaves_per_arg_: tuple[Any, ...]) -> Array:
        new_args = list(args)
        for arg_i_, (leaves_, treedef_, mask), float_leaves in zip(
            argnums, splits, float_leaves_per_arg_
        ):
            it = iter(float_leaves)
            merged = [next(it) if m else leaf for leaf, m in zip(leaves_, mask)]
            new_args[arg_i_] = jax.tree_util.tree_unflatten(treedef_, merged)
        return func(*new_args)

    float_leaves_per_arg = tuple(
        tuple(leaf for leaf, m in zip(leaves, mask) if m)
        for (leaves, _, mask) in splits
    )

    # jacfwd across the flat tuple of tuples of float leaves
    jac_float = jacfwd(call_from_float_leaves, argnums=tuple(range(len(argnums))))(
        *float_leaves_per_arg
    )

    # Recover output_shape from any produced Jacobian slab by stripping the trailing leaf-shape dims.
    out_shape: tuple[int, ...] | None = None
    for arg_i, float_jacs in enumerate(jac_float):
        for slab, leaf in zip(float_jacs, float_leaves_per_arg[arg_i]):
            out_shape = slab.shape[: slab.ndim - leaf.ndim]
            break
        if out_shape is not None:
            break
    assert out_shape is not None, (
        "cannot compute jacfwd Jacobian shape with no floating-point leaves"
    )

    # Splice zeros in for int leaves
    output = []
    for (leaves, treedef, mask), float_jacs in zip(splits, jac_float):
        merged_jacs: list[Array] = []
        float_iter = iter(float_jacs)
        for leaf, m in zip(leaves, mask):
            if m:
                merged_jacs.append(next(float_iter))
            else:
                merged_jacs.append(jnp.zeros(out_shape + leaf.shape, dtype=leaf.dtype))
        output.append(jax.tree_util.tree_unflatten(treedef, merged_jacs))
    return tuple(output)


def jacrev_custom(
    func: Callable[..., Array],
    jac_options: dict[str, Callable[[dict[str, Any]], Array] | None],
    n_profile_loops: int | None,
    func_name: str,
    static_argnames: Sequence[str] = (),
    mode: ADMode = "reverse",
    map_batch_size: int | None = None,
) -> Callable[
    ..., tuple[dict[str, Any], dict[str, float] | None, dict[str, float] | None]
]:
    r"""
    Obtain the Jacobians of the function `func` with respect to a chosen set of arguments.

    :param func: Function for which to obtain the Jacobians.
    :param jac_options: Dictionary with variable names as keys, with values being a tuple of the argument number in
    `func`, and an optional function which can be used to compute the given Jacobian. If this is None, the Jacobian is
    computed by applying AD with no approximations.
    :param n_profile_loops: Number of profile loops. If None, no profiling is done.
    :param func_name: Name of the function to be called, used for console prints during profiling.
    :param static_argnames: Argument names to treat as static when JIT-compiling the AD Jacobian. Needed so the
    profile loop hits a cached jaxpr instead of re-tracing on every call.
    :param mode: ``"reverse"`` (default) or ``"forward"``.
    :param map_batch_size: When set, batch passes to obtain Jacobian rather than vmapping, reducing memory at the
    expense of computation time.
    :return: Function to obtain Jacobians, as well as respective compile and run times for
    Jacobians if `n_profile_loops` is not None.
    """

    static_argnames_t = tuple(static_argnames)

    if mode == "reverse":
        if map_batch_size is not None:

            def _build_jac(argnames):
                return _jacrev_via_map_kwargs(
                    func, argnames=argnames, map_batch_size=map_batch_size
                )
        else:

            def _build_jac(argnames):
                return jacrev_kwargs(func, argnames=argnames, allow_int=True)
    elif mode == "forward":
        if map_batch_size is not None:

            def _build_jac(argnames):
                return _jacfwd_via_map_kwargs(
                    func, argnames=argnames, map_batch_size=map_batch_size
                )
        else:

            def _build_jac(argnames):
                return jacfwd_kwargs(func, argnames=argnames)
    else:
        raise ValueError('Invalid mode, use "forward" or "reverse"')

    def inner_func(
        **args: Any,
    ) -> tuple[dict[str, Any], dict[str, float] | None, dict[str, float] | None]:
        jacobians: dict[str, Any] = {}
        ad_args: list[
            str
        ] = []  # accumulate list of argument names that we will use for AD
        compile_time: dict[str, float | None] = {}
        run_time: dict[str, float | None] = {}

        # compute cases which have a passed approximation function for finding the Jacobian
        for arg, jac_func in jac_options.items():
            if jac_func is not None:
                jacobians[arg], compile_time[arg], run_time[arg] = conditional_profile(
                    func=jac_func,
                    n_loops=n_profile_loops,
                    func_name=func_name,
                    arg_name=arg,
                )(args)  # evaluate function
            else:
                ad_args.append(arg)  # indicate that we will use AD to compute

        # compute cases where we perform AD directly
        if ad_args:
            per_arg_jacobians: dict[str, Any] = {}
            if n_profile_loops is not None:
                for ad_arg in ad_args:
                    # profile for individual Jacobians
                    jac_func = jax.jit(
                        _build_jac(ad_arg),
                        static_argnames=static_argnames_t,
                    )
                    val, compile_time[ad_arg], run_time[ad_arg] = conditional_profile(
                        func=jac_func,
                        n_loops=n_profile_loops,
                        func_name=func_name,
                        arg_name=ad_arg,
                    )(**args)
                    per_arg_jacobians.update(val)

            if mode == "forward":
                all_jacobians = (
                    per_arg_jacobians
                    if n_profile_loops is not None
                    else _build_jac(ad_args)(**args)
                )
            else:
                jac_func = _build_jac(ad_args)
                if n_profile_loops is not None:
                    jac_func = jax.jit(jac_func, static_argnames=static_argnames_t)

                (
                    all_jacobians,
                    compile_time["all"],
                    run_time["all"],
                ) = conditional_profile(
                    func=jac_func,
                    n_loops=n_profile_loops,
                    func_name=func_name,
                    arg_name="all",
                )(**args)

            jacobians.update(all_jacobians)

        if n_profile_loops is not None:
            return jacobians, compile_time, run_time  # type: ignore
        else:
            return jacobians, None, None

    return inner_func


def _split_dyn_static(
    args: dict[str, Any], keep_dynamic: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""
    Split args into static and dynamic so that we can apply closure over the static arguments.
    """
    dynamic: dict[str, Any] = {}
    static: dict[str, Any] = {}
    keep = set(keep_dynamic)
    for name, val in args.items():
        if name in keep:
            dynamic[name] = val
            continue
        leaves = jax.tree.leaves(val)
        if any(hasattr(leaf, "shape") and hasattr(leaf, "dtype") for leaf in leaves):
            dynamic[name] = val
        else:
            static[name] = val
    return dynamic, static


def jacobian_approximation(
    func: Callable[..., Any],
    args: Any,
    approx_type: Literal["zero", "constant", "identity", "dense_linear", "lazy_linear"]
    | None,
    jacobian_argname: str,
    hessian_argnames: str | Sequence[str],
) -> Callable[..., Any] | None:
    r"""
    Compute approximations of Jacobians. The following options for approximation are available:
    `None` - No approximation is computed.
    `zero` - Assume that the Jacobian is zero.
    `identity` - Assume that the Jacobian is the identity matrix. Valid only for 2D square entries.
    `constant` - Assume that the Jacobian is constant across all time steps.
     Alongside these, options are also available to compute the Jacobians by a linear approximation by using a
     Hessian-vector product, where the Hessian is constant. This introduces additional options:
     `dense_linear` - The dense Hessian is explicitly computed, which increases memory cost but reduces computation cost.
     `lazy_linear` - Uses the JAX linearise routine, which avoids explicitly computing the dense Hessian, which reduces
     memory cost at the expense of increased computation cost.
    :param func: Function for which Jacobians are to be computed.
    :param args: Arguments for `func` which define the state around which we want to make the approximation.
    :param approx_type: Type of approximation to be used to compute Jacobians. Options are "constant", "dense_linear" or
    "lazy_linear". If None, no approximation is computed.
    :param jacobian_argname: Argument which the function Jacobian is to be computed with respect to.
    :param hessian_argnames: Argument names which the Jacobian is linearised with respect to for computing
    Hessian-vector products.
    :return: Function which approximates the Jacobian which takes the same arguments as `func`, or None.
    """

    hessian_argnames_t = (
        (hessian_argnames,)
        if isinstance(hessian_argnames, str)
        else tuple(hessian_argnames)
    )

    # closure over non-array arguments
    _dyn_args, _static_args = _split_dyn_static(
        args, keep_dynamic=(jacobian_argname, *hessian_argnames_t)
    )
    _func = func if not _static_args else (lambda **_dk: func(**_dk, **_static_args))

    match approx_type:
        case None:
            # no approximation, and so we will compute the exact Jacobian using AD later
            return None
        case "zero":
            # assume that the Jacobian is zero, avoiding any Jacobian computation. In practice, this is only useful
            # when the `constant` path would be slow to linearise.
            jac_shape = jax.eval_shape(
                jacrev_kwargs(_func, argnames=jacobian_argname, allow_int=True),
                **_dyn_args,
            )[jacobian_argname]
            zero_jac = jax.tree.map(
                lambda s: jnp.zeros(s.shape, dtype=s.dtype), jac_shape
            )
            return lambda *_, **__: zero_jac
        case "identity":
            # assume that the Jacobian is the identity matrix
            jac_shape = jax.eval_shape(
                jacrev_kwargs(_func, argnames=jacobian_argname, allow_int=True),
                **_dyn_args,
            )[jacobian_argname]
            if not isinstance(jac_shape, jax.ShapeDtypeStruct):
                raise TypeError(
                    "identity approximation requires an Array-valued Jacobian, got a pytree"
                )
            shape = jac_shape.shape
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(
                    f"identity approximation expected a 2-D square Jacobian for "
                    f"{jacobian_argname}, got shape {shape}"
                )
            identity_jac = jnp.eye(shape[0], dtype=jac_shape.dtype)
            return lambda *_, **__: identity_jac
        case "constant":
            # assume that the Jacobian stays constant for all time
            jac = jacrev_kwargs(_func, argnames=jacobian_argname, allow_int=True)(
                **_dyn_args
            )[jacobian_argname]
            return lambda *_, **__: jac
        case "lazy_linear":
            # Jacobian is computed using a Hessian-vector product where the Hessian is chosen to be constant
            # uses the JAX linearise function which means that the dense Hessian is never computed
            # reduced memory cost at the expense of increased computation cost
            def _jac_at(*y_vals: Any) -> Array:
                new_args = dict(_dyn_args)
                for name, val in zip(hessian_argnames_t, y_vals):
                    new_args[name] = val
                return jacrev_kwargs(_func, argnames=jacobian_argname, allow_int=True)(
                    **new_args
                )[jacobian_argname]

            y0_vals = tuple(_dyn_args[name] for name in hessian_argnames_t)
            j0, jac_jvp = jax.linearize(_jac_at, *y0_vals)

            def lazy_linear_approx(
                new_args: dict[str, Any], *_: Any, **__: Any
            ) -> Array:
                dy_vals = tuple(
                    new_args[name] - _dyn_args[name] for name in hessian_argnames_t
                )
                return j0 + jac_jvp(*dy_vals)

            return lazy_linear_approx
        case "dense_linear":
            # as `lazy_linear`, except that the dense Hessian is explicitly computed.
            def _jac_at_kwargs(**kw: Any) -> Array:
                new_args = dict(_dyn_args)
                new_args.update(kw)
                return jacrev_kwargs(_func, argnames=jacobian_argname, allow_int=True)(
                    **new_args
                )[jacobian_argname]

            y0_kwargs = {name: _dyn_args[name] for name in hessian_argnames_t}
            j0 = _jac_at_kwargs(**y0_kwargs)
            hessians = jacrev_kwargs(
                _jac_at_kwargs, argnames=hessian_argnames_t, allow_int=True
            )(**y0_kwargs)

            def dense_linear_approx(
                new_args: dict[str, Any], *_: Any, **__: Any
            ) -> Array:
                result = j0
                for name in hessian_argnames_t:
                    dy = new_args[name] - _dyn_args[name]
                    result += jnp.tensordot(hessians[name], dy, axes=dy.ndim)
                return result

            return dense_linear_approx
        case _:
            raise ValueError(f"Invalid approximation type: {approx_type}")


def construct_approximation(
    res_args: dict[str, tuple[Callable[..., Any], dict, Sequence[str]]],
    jacobian_approximations: AeroJacobianApproximations | BeamJacobianApproximations,
) -> dict[str, dict[str, Callable[..., Any] | None]]:
    jacobian_options: dict[str, dict[str, Callable[..., Any] | None]] = {}
    for res_name, (func, args, diff_arg_names) in res_args.items():
        approx_class = getattr(jacobian_approximations, res_name)
        inner: dict[str, Callable[..., Any] | None] = {k: None for k in diff_arg_names}
        for arg_name, entry in asdict(approx_class).items():
            if entry is None:
                inner[arg_name] = None
            elif entry in ("zero", "constant", "identity"):
                inner[arg_name] = jacobian_approximation(
                    func=func,
                    args=args,
                    approx_type=entry,
                    jacobian_argname=arg_name,
                    hessian_argnames=(),
                )
            elif isinstance(entry, tuple) and len(entry) == 2:
                approx_type, hessian_argnames = entry
                inner[arg_name] = jacobian_approximation(
                    func=func,
                    args=args,
                    approx_type=approx_type,
                    jacobian_argname=arg_name,
                    hessian_argnames=hessian_argnames,
                )
            else:
                raise NotImplementedError("Invalid Jacobian approx type")
        jacobian_options[res_name] = inner
    return jacobian_options
