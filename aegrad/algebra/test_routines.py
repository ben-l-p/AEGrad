from jax import Array, vmap
import jax.numpy as jnp
from math import factorial
from jax.scipy.special import bernoulli
from aegrad.algebra.constants import BASE_SUMMATION_ORDER
from typing import Sequence, Literal


def check_if_so3_g(
    rmat: Array,
    raise_if_false: bool = True,
) -> bool:
    r"""
    Check if rotation matrix is a valid SO3 group element
    :param rmat: Rotation matrix, [3, 3]
    :param raise_if_false: If the check fails, raise ValueError
    :return: Boolean indicating if matrix is SO3
    """
    column_mags = jnp.linalg.norm(rmat, axis=0)
    row_mags = jnp.linalg.norm(rmat, axis=1)

    # if not correct shapes
    if rmat.shape != (3, 3):
        if raise_if_false:
            raise ValueError("Matrix not SO3 as shapes is not (3, 3)")
        return False

    # if not unit magnitude
    if not jnp.allclose(jnp.concatenate((column_mags, row_mags)), 1.0):
        if raise_if_false:
            raise ValueError(
                "Matrix is not SO3 as rows or columns are not of unit magnitude"
            )
        return False

    # if not othogonal
    if not (
        jnp.allclose(rmat.T @ rmat, jnp.eye(3))
        and jnp.allclose(rmat @ rmat.T, jnp.eye(3))
    ):
        if raise_if_false:
            raise ValueError("Matrix is not SO3 as it is not orthogonal")
        return False
    return True


def check_if_so3_a(h_tilde: Array, raise_if_false: bool = True) -> bool:
    r"""
    Check if rotation matrix is a valid so3 algebra element
    :param h_tilde: Algebra matrix, [3, 3]
    :param raise_if_false: If the check fails, raise ValueError
    :return: Boolean indicating if matrix is so3
    """
    # check shapes
    if h_tilde.shape != (3, 3):
        if raise_if_false:
            raise ValueError("Matrix not so3 as shapes is not (3, 3)")
        return False

    # check for nonzero diagonal elements
    if jnp.any(jnp.diagonal(h_tilde)) != 0.0:
        if raise_if_false:
            raise ValueError("Matrix not so3 as diagonal elements are not zero")
        return False

    # check for skew symmetry
    if jnp.any(h_tilde + h_tilde.T):
        if raise_if_false:
            raise ValueError("Matrix not so3 as it is not skew symmetric")
        return False
    return True


def check_if_se3_g(hg: Array, raise_if_false: bool = True) -> bool:
    r"""
    Check if matrix is a valid SE3 group element
    :param hg: SE(3) matrix, [4, 4]
    :param raise_if_false: If the check fails, raise ValueError
    :return: Boolean indicating if matrix is SE(3)
    """
    # check shapes
    if hg.shape != (4, 4):
        if raise_if_false:
            raise ValueError("Matrix not SE3 as shapes is not (4, 4)")
        return False

    # check rotational component
    if not check_if_so3_g(hg[:3, :3], raise_if_false=raise_if_false):
        return False

    # check last row
    if not jnp.allclose(hg[3, :], jnp.array([0.0, 0.0, 0.0, 1.0])):
        if raise_if_false:
            raise ValueError("Matrix not SE3 as last row is not [0, 0, 0, 1]")
        return False
    return True


def check_if_all_se3_g(hgs: Array, raise_if_false: bool = True) -> bool:
    r"""
    Check if array of matrices are valid SE3 group elements
    :param hgs: SE(3) matrices, [..., 4, 4]
    :param raise_if_false: If the check fails, raise ValueError
    :return: Boolean indicating if all matrices are SE(3)
    """
    # check shape
    if hgs.shape[-2:] != (4, 4):
        if raise_if_false:
            raise ValueError("Input not se3 as last two dimensions are not (4, 4)")
        return False

    def check_if_so3_g_jittable(rmat: Array) -> Array:
        column_mags = jnp.linalg.norm(rmat, axis=0)
        row_mags = jnp.linalg.norm(rmat, axis=1)

        # check if unit magnitude
        out = jnp.all(jnp.allclose(jnp.concatenate((column_mags, row_mags)), 1.0))

        # check if orthogonal
        out &= jnp.all(jnp.allclose(rmat.T @ rmat, jnp.eye(3), atol=1e-5, rtol=1e-3))
        out &= jnp.all(jnp.allclose(rmat @ rmat.T, jnp.eye(3), atol=1e-5, rtol=1e-3))
        return out

    hgs_flat = hgs.reshape(-1, 4, 4)

    results = jnp.all(
        vmap(check_if_so3_g_jittable, in_axes=0, out_axes=0)(hgs_flat[:, :3, :3])
    )
    results &= jnp.all(jnp.allclose(hgs_flat[:, 3, :3], 0.0))
    results &= jnp.all(jnp.allclose(hgs_flat[:, 3, 3], 1.0))

    if not results:
        if raise_if_false:
            raise ValueError("Not all matrices are se3 elements")
        return False
    return True


def check_if_se3_a(h_tilde: Array, raise_if_false: bool = True) -> bool:
    r"""
    Check if matrix is a valid se(3) group element
    :param h_tilde: se(3) matrix, [4, 4]
    :param raise_if_false: If the check fails, raise ValueError
    :return: Boolean indicating if matrix is se(3)
    """
    # check shapes
    if h_tilde.shape != (4, 4):
        if raise_if_false:
            raise ValueError("Matrix not se3 as shapes is not (4, 4)")
        return False

    # check so3 component
    if not check_if_so3_a(h_tilde[:3, :3], raise_if_false=raise_if_false):
        return False

    # check last row and column
    if jnp.any(h_tilde[3, :]):
        if raise_if_false:
            raise ValueError("Matrix not se3 as last row is not zero")
        return False
    return True


def check_if_all_se3_a(h_tildes: Array, raise_if_false: bool = True) -> bool:
    r"""
    Check if array of matrices are valid se(3) algebra elements
    :param h_tildes: se(3) matrices, [..., 4, 4]
    :param raise_if_false: If the check fails, raise ValueError
    :return: Boolean indicating if all matrices are se(3)
    """
    h_tildes_flat = h_tildes.reshape(-1, 4, 4)

    # check bottom row
    results = jnp.all(jnp.allclose(h_tildes_flat[:, 3, :], 0.0))

    # check diagonal
    results &= jnp.all(jnp.allclose(h_tildes_flat[:, (0, 1, 2), (0, 1, 2)], 0.0))

    # check skew symmetry of so3 part
    results &= jnp.all(
        jnp.allclose(
            h_tildes_flat[:, :3, :3],
            -jnp.transpose(h_tildes_flat[:, :3, :3], (0, 2, 1)),
        )
    )

    if not results:
        if raise_if_false:
            raise ValueError("Not all matrices are se3 algebra elements")
        return False
    return True


def exp_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the matrix exponential using truncated summation. This is used to validate other implementations.
    :param a: Algebra matrix to exponentiate, [n, n]
    :param order: Order of summation.
    :return: Exponential of matrix, [n, n]
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += jnp.linalg.matrix_power(a, i) / factorial(i)
    return result


def log_sum(g: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the matrix logarithm using truncated summation. This is used to validate other implementations.
    :param g: Group matrix to exponentiate, [n, n]
    :param order: Order of summation.
    :return: Logarithm of matrix, [n, n]
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
    :param a: Adjoint action matrix, [n, n]
    :param order: Order of summation.
    :return: Tangent operator, [n, n]
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += (-1.0) ** i * jnp.linalg.matrix_power(a, i) / factorial(i + 1)
    return result


def t_inv_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the inverse tangent operator truncated summation. This is used to validate other implementations.
    :param a: Adjoint action matrix, [n, n]
    :param order: Order of summation.
    :return: Inverse angent operator, [n, n]
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    b = bernoulli(order)

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += (-1.0) ** i * b[i] * jnp.linalg.matrix_power(a, i) / factorial(i)
    return result


def k_t_expected(coeffs: Array | Sequence[float], l: Array | float) -> Array:
    r"""
    Compute expected two-node beam undeformed element stiffness matrix given coefficients and length
    :param coeffs: Stiffness coefficients which make up the diagonal of the local stiffness matrix, [6]
    :param l: Beam length, []
    :return: Beam tangent stiffness matrix, [12, 12]
    """
    if (isinstance(coeffs, Array) and coeffs.shape != (6,)) or (
        isinstance(coeffs, Sequence) and len(coeffs) != 6
    ):
        raise ValueError("Coefficients array must be of shapes (6, )")

    if isinstance(l, Array) and not jnp.isscalar(l):
        raise ValueError("Length l must be a scalar value")

    eax, gay, gaz, gjx, eiy, eiz = coeffs

    k_upper_left = jnp.array(
        [
            [eax / l, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 12.0 * eiz / l**3, 0.0, 0.0, 0.0, 6.0 * eiz / l**2],
            [0.0, 0.0, 12.0 * eiy / l**3, 0.0, -6.0 * eiy / l**2, 0.0],
            [0.0, 0.0, 0.0, gjx / l, 0.0, 0.0],
            [0.0, 0.0, -6.0 * eiy / l**2, 0.0, 4.0 * eiy / l, 0.0],
            [0.0, 6.0 * eiz / l**2, 0.0, 0.0, 0.0, 4.0 * eiz / l],
        ]
    )

    k_upper_right = jnp.array(
        [
            [-eax / l, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -12.0 * eiz / l**3, 0.0, 0.0, 0.0, 6.0 * eiz / l**2],
            [0.0, 0.0, -12.0 * eiy / l**3, 0.0, -6.0 * eiy / l**2, 0.0],
            [0.0, 0.0, 0.0, -gjx / l, 0.0, 0.0],
            [0.0, 0.0, 6.0 * eiy / l**2, 0.0, 2.0 * eiy / l, 0.0],
            [0.0, -6.0 * eiz / l**2, 0.0, 0.0, 0.0, 2.0 * eiz / l],
        ]
    )

    k_lower_left = k_upper_right.T

    k_lower_right = k_upper_left
    k_lower_right = k_lower_right.at[1:3, 4:6].mul(-1.0)
    k_lower_right = k_lower_right.at[4:6, 1:3].mul(-1.0)

    return jnp.block([[k_upper_left, k_upper_right], [k_lower_left, k_lower_right]])


def const_curvature_beam(
    kappa: float | Array, s: float | Array, direction: Literal["y", "z"]
) -> Array:
    r"""
    For a beam with constant curvature, with base node at the origin and curvature in the positive z direction
    (i.e., existing in the x-y plane with z=0), obtain the coordinates along the beam length for
    :param kappa: Curvature of the element, []
    :param s: Position along the beam length, []
    :param direction: Direction of moment applied, either 'y' or 'z'
    :return: Coordinate of point along the beam, [3]
    """

    x = jnp.sin(s * kappa) / kappa
    v_defl = (1.0 - jnp.cos(s * kappa)) / kappa

    match direction:
        case "y":
            return jnp.array([x, 0.0, -v_defl])
        case "z":
            return jnp.array([x, v_defl, 0.0])
        case _:
            raise ValueError("Direction must be 'y' or 'z'")


def get_curvature(d: Array) -> Array:
    r"""
    Obtain curvature from relative configuration vector
    :param d: Relative configuration vector, [6]
    :return: Curvature of neutral axis, []
    """
    return jnp.linalg.norm(jnp.cross(d[3:], d[:3])) / jnp.linalg.norm(d[:3])


def get_torsion(d: Array) -> Array:
    r"""
    Obtain torsion from relative configuration vector
    :param d: Relative configuration vector, [6]
    :return: Torsion of neutral axis, []
    """
    return -jnp.inner(d[3:], d[:3]) / jnp.linalg.norm(d[:3])
