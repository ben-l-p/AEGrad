import jax
from jax import numpy as jnp

from algebra.base import clip_to_pi, chi
from constants import SMALL_ANG_THRESH
from algebra.so3 import (
    vec_to_skew,
    skew_to_vec,
    exp_so3,
    log_so3,
    t_so3,
    t_inv_so3,
    bound_h_omega,
    bracket_so3,
)
from algebra.se3 import (
    ha_to_ha_hat,
    ha_hat_to_ha,
    ha_to_ha_tilde,
    ha_tilde_to_ha,
    exp_se3,
    log_se3,
    hg_inv,
    x_rmat_to_hg,
    hg_to_x_rmat,
    t_u_omega_plus,
    t_u_omega_minus,
    t_se3,
    t_inv_se3,
    bracket_se3,
    t_star,
    hg_to_ha_hat,
    ha_to_ha_check,
)
from algebra.test_routines import (
    check_if_so3_a,
    check_if_so3_g,
    check_if_se3_a,
    check_if_se3_g,
)
from algebra.base import exp_sum, log_sum, t_sum, t_inv_sum

jax.config.update("jax_enable_x64", True)


class TestConstants:
    @staticmethod
    def test_small_ang_thresh():
        assert SMALL_ANG_THRESH > 0.0, (
            f"Small angle threshold should be greater than 0, but got {SMALL_ANG_THRESH}"
        )


class TestTilde:
    r"""
    Test functions involving skew symmetric matrices
    """

    @staticmethod
    def test_so3_identity():
        skew = vec_to_skew(jnp.zeros(3))
        check_if_so3_a(skew)
        assert jnp.allclose(skew, 0.0), (
            f"Skew symmetric of zero vector should yield zero matrix, returned {skew}"
        )

    @staticmethod
    def test_so3_inverse():
        vec = jnp.linspace(1.0, 3.0, 3)
        skew = vec_to_skew(vec)
        check_if_so3_a(skew)
        assert jnp.allclose(out := skew_to_vec(skew), vec), (
            f"Inverse transformation between vector and skew "
            f"should yield original vector {vec}, returned {out}"
        )

    @staticmethod
    def test_so3_tilde_times_vec():
        vec = jnp.linspace(1.0, 3.0, 3)
        skew = vec_to_skew(vec)
        check_if_so3_a(skew)
        assert jnp.allclose(out := skew @ vec, 0.0), (
            f"Product of skew symmetric matrix and its vector "
            f"should yield zero, returned {out}"
        )

    @staticmethod
    def test_se3_identity():
        ha_tilde = ha_to_ha_tilde(jnp.zeros(6))
        check_if_se3_a(ha_tilde)
        assert jnp.allclose(ha_tilde, 0.0), (
            f"Lie algebra of zero vector should yield zero matrix, returned {ha_tilde}"
        )

    @staticmethod
    def test_se3_inverse():
        vec = jnp.linspace(1.0, 6.0, 6)
        ha_tilde = ha_to_ha_tilde(vec)
        check_if_se3_a(ha_tilde)
        assert jnp.allclose(out := ha_tilde_to_ha(ha_tilde), vec), (
            f"Inverse transformation of transformation should yield original vector {vec}, returned {out} "
        )

    @staticmethod
    def test_se3_hat_identity():
        assert jnp.allclose(out := ha_to_ha_hat(jnp.zeros(6)), 0.0), (
            f"Adjoint of zero vector should yield zero matrix, returned {out}"
        )

    @staticmethod
    def test_se3_hat_inverse():
        vec = jnp.linspace(1.0, 6.0, 6)
        assert jnp.allclose(out := ha_hat_to_ha(ha_to_ha_hat(vec)), vec), (
            f"Inverse transformation of hat should yield original vector {vec}, returned {out} "
        )

    @staticmethod
    def test_se3_hat_times_vec():
        vec = jnp.linspace(1.0, 6.0, 6)
        assert jnp.allclose(out := ha_to_ha_hat(vec) @ vec, 0.0), (
            f"Product of lie algebra and its vector should yield zero, returned {out}"
        )

    @staticmethod
    def test_hg_se3_inverse_identity():
        hg_inv_ = hg_inv(jnp.eye(4))
        check_if_se3_g(hg_inv_)
        assert jnp.allclose(hg_inv_, jnp.eye(4)), (
            f"Inverse of identity should yield identity, returned {hg_inv_}"
        )

    @staticmethod
    def test_hg_se3_inverse():
        vec = jnp.linspace(0.6, 0.1, 6)
        hg = exp_se3(vec)
        check_if_se3_g(hg)
        hg_inv_ = hg_inv(hg)
        check_if_se3_g(hg_inv_)
        x, rmat = hg_to_x_rmat(hg)
        check_if_so3_g(rmat)
        assert jnp.allclose(out := hg @ hg_inv_, jnp.eye(4)), (
            f"Product of SE3 matrix and its inverse should yield "
            f"zero matrix, returned {out}"
        )
        assert jnp.allclose(out := hg_inv_ @ hg, jnp.eye(4)), (
            f"Product of SE3 matrix inverse and original matrix should yield "
            f"zero matrix, returned {out}"
        )

    @staticmethod
    def test_x_rmat():
        x = jnp.linspace(0.4, 0.6, 3)
        crv = jnp.linspace(0.1, 0.3, 3)
        rmat = exp_so3(crv)
        check_if_so3_g(rmat)
        hg = x_rmat_to_hg(x, rmat)
        check_if_se3_g(hg)
        x_out, rmat_out = hg_to_x_rmat(hg)
        check_if_so3_g(rmat_out)
        assert jnp.allclose(rmat_out, rmat), (
            f"Input rotation matrix should match output, in {rmat}, out {rmat_out}"
        )
        assert jnp.allclose(x_out, x), (
            f"Input vector should match output, in {x}, out {x_out}"
        )


class TestUtils:
    r"""
    Test functions involving angle normalisation
    """

    @staticmethod
    def clip_to_pi_within_range():
        angs = [-0.5 * jnp.pi, 0.0, 0.5 * jnp.pi]
        clipped_angs = [clip_to_pi(ang) for ang in angs]
        for ang, clipped_ang in zip(angs, clipped_angs):
            assert ang == clipped_ang, (
                f"Angle clipping values in range should not affect result, in={ang}, out={clipped_ang}"
            )

    @staticmethod
    def clip_to_pi_outside_range():
        angs = [-1.5 * jnp.pi, 1.5 * jnp.pi]
        clipped_angs = [clip_to_pi(ang) for ang in angs]
        expected_angs = [0.5 * jnp.pi, -0.5 * jnp.pi]
        for ang, clipped_ang in zip(expected_angs, clipped_angs):
            assert jnp.isclose(ang, clipped_ang), (
                f"Angle clipping values in range should not affect result, in={ang}, out={clipped_ang}"
            )

    @staticmethod
    def clip_rot_vec_within_range():
        vecs = [jnp.zeros(3).at[i].set(0.5) for i in range(3)]
        clipped_vecs = [bound_h_omega(vec) for vec in vecs]
        for vec, clipped_vec in zip(vecs, clipped_vecs):
            assert jnp.allclose(vec, clipped_vec), (
                f"Angle clipping vectors in range should not affect result, in={vec}, out={clipped_vec}"
            )

    @staticmethod
    def clip_rot_vec_outside_range():
        vecs = [
            jnp.zeros(3).at[i].set(s * 1.5 * jnp.pi)
            for i in range(3)
            for s in (-1.0, 1.0)
        ]
        expected_vecs = [
            jnp.zeros(3).at[i].set(-s * 0.5 * jnp.pi)
            for i in range(3)
            for s in (-1.0, 1.0)
        ]
        clipped_vecs = [bound_h_omega(vec) for vec in vecs]
        for vec, clipped_vec in zip(expected_vecs, clipped_vecs):
            assert jnp.allclose(vec, clipped_vec), (
                f"Angle clipping vectors in range should not affect result, in={vec}, out={clipped_vec}"
            )

    @staticmethod
    def test_chi():
        vec = jnp.linspace(0.1, 0.3, 3)
        rmat = exp_so3(vec)
        check_if_so3_g(rmat)
        chi_ = chi(rmat)

        assert jnp.allclose(chi_.T @ chi_, jnp.eye(6)), (
            f"Chi matrix should be orthogonal and premultiplying by its "
            f"transpose should return the identity, returned {chi_}"
        )
        assert jnp.allclose(chi_ @ chi_.T, jnp.eye(6)), (
            f"Chi matrix should be orthogonal and premultiplying by its "
            f"transpose should return the identity, returned {chi_}"
        )


class TestBracket:
    @staticmethod
    def test_bracket_so3():
        vec1 = jnp.linspace(0.1, 0.3, 3)
        vec2 = jnp.linspace(0.4, 0.6, 3)

        assert jnp.allclose(
            lhs := bracket_so3(vec1, vec2), rhs := -bracket_so3(vec2, vec1)
        ), f"so3 bracket operator identity not met, expected {rhs}, returned {lhs}"

    @staticmethod
    def test_bracket_se3():
        vec1 = jnp.linspace(0.1, 0.6, 6)
        vec2 = jnp.linspace(0.7, 1.2, 6)

        assert jnp.allclose(
            lhs := bracket_se3(vec1, vec2), rhs := -bracket_se3(vec2, vec1)
        ), f"se3 bracket operator identity not met, expected {rhs}, returned {lhs}"


class TestTangent:
    r"""
    Test functions involving the tangent operator
    """

    @staticmethod
    def test_so3_identity():
        assert jnp.allclose(out := t_so3(jnp.zeros(3)), jnp.eye(3)), (
            f"SO3 tangent at zero should yield identity matrix., returned {out}"
        )

    @staticmethod
    def test_so3_inverse_identity():
        assert jnp.allclose(out := t_inv_so3(jnp.zeros(3)), jnp.eye(3)), (
            f"SO3 inverse tangent at zero should yield identity matrix, returned {out}"
        )

    @staticmethod
    def test_so3_inverse():
        vec = jnp.linspace(0.1, 0.3, 3)
        assert jnp.allclose(out := t_inv_so3(vec), inv := jnp.linalg.inv(t_so3(vec))), (
            f"SO3 inverse tangent should equal matrix "
            f"inverse of tangent {inv}, returned {out}"
        )

    @staticmethod
    def test_so3_finite():
        vec = jnp.linspace(0.1, 0.3, 3)
        assert jnp.allclose(out := t_inv_so3(vec) @ t_so3(vec), jnp.eye(3)), (
            f"Product of tangent and its inverse should yield "
            f"identity matrix, returned {out}"
        )
        assert jnp.allclose(out := t_so3(vec) @ t_inv_so3(vec), jnp.eye(3)), (
            f"Product of tangent and its inverse should yield "
            f"identity matrix, returned {out}"
        )

    @staticmethod
    def test_so3_summation_series():
        vec = jnp.linspace(0.1, 0.3, 3)
        assert jnp.allclose(out := t_so3(vec), sum_ := t_sum(vec_to_skew(vec))), (
            f"SO3 tangent should equal summation series approximations {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_so3_inv_summation_series():
        vec = jnp.linspace(0.1, 0.3, 3)
        assert jnp.allclose(
            out := t_inv_so3(vec), sum_ := t_inv_sum(vec_to_skew(vec))
        ), (
            f"SO3 tangent should equal summation series approximation {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_se3_inverse():
        vec = jnp.linspace(0.1, 0.6, 6)
        assert jnp.allclose(out := t_inv_se3(vec), inv := jnp.linalg.inv(t_se3(vec))), (
            f"SE3 inverse tangent should equal matrix "
            f"inverse of tangent {inv}, returned {out}"
        )

    @staticmethod
    def test_se3_finite():
        vec = jnp.linspace(0.1, 0.6, 6)
        assert jnp.allclose(out := t_inv_se3(vec) @ t_se3(vec), jnp.eye(6)), (
            f"Product of tangent and its inverse should yield "
            f"identity matrix, returned {out}"
        )
        assert jnp.allclose(out := t_se3(vec) @ t_inv_se3(vec), jnp.eye(6)), (
            f"Product of tangent and its inverse should yield "
            f"identity matrix, returned {out}"
        )

    @staticmethod
    def test_se3_summation_series():
        vec = jnp.linspace(0.1, 0.6, 6)
        assert jnp.allclose(out := t_se3(vec), sum_ := t_sum(ha_to_ha_hat(vec))), (
            f"SE3 tangent should equal summation series approximations {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_se3_inv_summation_series():
        vec = jnp.linspace(0.1, 0.6, 6)
        assert jnp.allclose(
            out := t_inv_se3(vec), sum_ := t_inv_sum(ha_to_ha_hat(vec))
        ), (
            f"SE3 tangent should equal summation series approximation {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_se3_u_omega_plus_zero():
        assert jnp.allclose(out := t_u_omega_plus(jnp.zeros(6)), 0.0), (
            f"T_u_omega_plus should return zero for zero input, returned {out}"
        )

    @staticmethod
    def test_se3_u_omega_plus_zero_rot():
        vec = jnp.array((0.1, 0.2, 0.3, 0.0, 0.0, 0.0))
        assert jnp.allclose(
            out := t_u_omega_plus(vec), val := -0.5 * vec_to_skew(vec[:3])
        ), (
            f"T_u_omega_plus should return {val} for zero rotation component, returned {out}"
        )

    @staticmethod
    def test_se3_u_omega_minus_zero():
        assert jnp.allclose(out := t_u_omega_minus(jnp.zeros(6)), 0.0), (
            f"T_u_omega_minus should return zero for zero input, returned {out}"
        )

    @staticmethod
    def test_se3_u_omega_minus_zero_rot():
        vec = jnp.array((0.1, 0.2, 0.3, 0.0, 0.0, 0.0))
        assert jnp.allclose(
            out := t_u_omega_minus(vec), val := 0.5 * vec_to_skew(vec[:3])
        ), (
            f"T_u_omega_minus should return {val} for zero rotation component, returned {out}"
        )

    @staticmethod
    def test_t_star_unity():
        d = jnp.linspace(0.1, 0.6, 6)
        assert jnp.allclose(out := t_star(jnp.ones(1), d), jnp.eye(6)), (
            f"T_star with unity scalar should return identity matrix, returned {out}"
        )


class TestExpLog:
    r"""
    Test functions involving the matrix exponential and logarithm operator
    """

    @staticmethod
    def test_exp_so3_identity():
        rmat = exp_so3(jnp.zeros(3))
        check_if_so3_g(rmat)
        assert jnp.allclose(rmat, jnp.eye(3)), (
            f"SO3 exponential at zero should yield identity matrix, returned {rmat}"
        )

    @staticmethod
    def test_log_so3_identity():
        assert jnp.allclose(out := log_so3(jnp.eye(3)), 0.0), (
            f"SO3 logarithm at identity should yield zero, returned {out}"
        )

    @staticmethod
    def test_so3_finite():
        vec = jnp.linspace(0.1, 0.3, 3)
        rmat = exp_so3(vec)
        check_if_so3_g(rmat)
        assert jnp.allclose(out := log_so3(rmat), vec), (
            f"SO3 Logarithm of exponential of vector should return same vector "
            f"{vec}, returned {out}"
        )

    @staticmethod
    def test_exp_so3_summation_series():
        vec = jnp.linspace(0.1, 0.3, 3)
        assert jnp.allclose(out := exp_so3(vec), sum_ := exp_sum(vec_to_skew(vec))), (
            f"SO3 exponential should equal summation series approximation {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_log_so3_summation_series():
        vec = jnp.linspace(0.1, 0.3, 3)
        rmat = exp_so3(vec)
        check_if_so3_g(rmat)
        assert jnp.allclose(out := log_so3(rmat), sum_ := skew_to_vec(log_sum(rmat))), (
            f"SO3 logarithm should equal summation series approximation {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_exp_se3_identity():
        hg = exp_se3(jnp.zeros(6))
        check_if_se3_g(hg)
        assert jnp.allclose(hg, jnp.eye(4)), (
            f"SE3 exponential at zero should yield identity matrix, returned {hg}"
        )

    @staticmethod
    def test_log_se3_identity():
        assert jnp.allclose(out := log_so3(jnp.eye(4)), 0.0), (
            f"SE3 logarithm at identity should yield zero, returned {out}"
        )

    @staticmethod
    def test_se3_finite():
        vec = jnp.linspace(0.1, 0.6, 6)
        hg = exp_se3(vec)
        check_if_se3_g(hg)
        assert (out := log_se3(hg), vec), (
            f"SE3 Logarithm of exponential of vector should return same vector "
            f"{vec}, returned {out}"
        )

    @staticmethod
    def test_exp_se3_summation_series():
        vec = jnp.linspace(0.1, 0.6, 6)
        assert jnp.allclose(
            out := exp_se3(vec), sum_ := exp_sum(ha_to_ha_tilde(vec))
        ), (
            f"SE3 exponential should equal summation series approximation {sum_}, "
            f"returned {out}"
        )

    @staticmethod
    def test_log_se3_summation_series():
        vec = jnp.linspace(0.1, 0.6, 6)
        hg = exp_se3(vec)
        check_if_se3_g(hg)
        assert jnp.allclose(
            out := log_se3(hg), sum_ := ha_tilde_to_ha(log_sum(hg)), rtol=1e-2
        ), (
            f"SO3 logarithm should equal summation series approximation {sum_}, "
            f"returned {out}"
        )


class TestCheckMatrix:
    @staticmethod
    def test_se3_check():
        h1 = jnp.linspace(0.1, 0.6, 6)
        h2 = jnp.linspace(0.7, 1.2, 6)

        lhs = ha_to_ha_hat(h1).T @ h2
        rhs = ha_to_ha_check(h2).T @ h1
        assert jnp.allclose(lhs, rhs), (
            f"ha_hat and ha_check are not consistent, lhs={lhs}, rhs={rhs}"
        )


class TestAdjoint:
    @staticmethod
    def test_se3_group_adjoint():
        vals = jnp.linspace(0.6, 0.1, 6)
        delta = jnp.linspace(0.05, 0.25, 6)
        hg = exp_se3(vals)

        d_tilde = ha_to_ha_tilde(delta)

        full_output = ha_tilde_to_ha(hg @ d_tilde @ hg_inv(hg))
        adjoint_output = hg_to_ha_hat(hg) @ delta

        assert jnp.allclose(full_output, adjoint_output), (
            f"SE3 adjoint operator failed, expected {adjoint_output}, returned {full_output}"
        )

    @staticmethod
    def test_se3_algebra_adjoint():
        vals = jnp.linspace(0.1, 0.6, 6)
        delta = jnp.full(6, 1e-3)

        full_output = ha_tilde_to_ha(bracket_se3(vals, delta))
        adjoint_output = ha_to_ha_hat(vals) @ delta

        assert jnp.allclose(full_output, adjoint_output), (
            f"SE3 adjoint operator failed, expected {adjoint_output}, returned {full_output}"
        )


class TestIdentitiesSO3:
    x = jnp.linspace(0.1, 0.3, 3)  # test R3 vector
    a = 0.75  # constant, result shouldn't depend on the choice

    # all formula are given in canonical form
    tilde = vec_to_skew  # [3] -> [3, 3]
    hat = vec_to_skew  # [3] -> [3, 3]
    exp = lambda x_: exp_so3(skew_to_vec(x_))  # [3, 3] -> [3, 3]
    log = lambda r_: vec_to_skew(log_so3(r_))  # [3, 3] -> [3, 3]
    t = t_so3  # [3] -> [3, 3]
    t_inv = t_inv_so3  # [3] -> [3, 3]

    @classmethod
    def test_c4(cls):
        assert jnp.allclose(lhs := cls.t(cls.a * cls.x) @ cls.x, rhs := cls.x), (
            f"{lhs} should be equal to {rhs}"
        )

    @classmethod
    def test_c5(cls):
        assert jnp.allclose(
            lhs := cls.t_inv(cls.x) - cls.t_inv(-cls.x), rhs := cls.hat(cls.x)
        ), f"{lhs} should be equal to {rhs}"

    @classmethod
    def test_c7(cls):
        assert jnp.allclose(
            lhs := cls.t(-cls.x) @ cls.t(cls.x), rhs := cls.t(cls.x) @ cls.t(-cls.x)
        ), f"{lhs} should be equal to {rhs}"

    @classmethod
    def test_c8(cls):
        assert jnp.allclose(
            lhs := cls.t_inv(-cls.x) @ cls.t(cls.x),
            rhs := cls.t(cls.x) @ cls.t_inv(-cls.x),
        ), f"{lhs} should be equal to {rhs}"

    @classmethod
    def test_c9(cls):
        assert jnp.allclose(
            lhs := cls.t(-cls.x) @ cls.t_inv(cls.x),
            rhs := cls.t_inv(cls.x) @ cls.t(-cls.x),
        ), f"{lhs} should be equal to {rhs}"


class TestIdentitySO3Identity(TestIdentitiesSO3):
    x = jnp.zeros(3)


class TestIdentitiesSE3(TestIdentitiesSO3):
    x = jnp.linspace(0.1, 0.6, 6)

    tilde = ha_to_ha_tilde  # [6] -> [4, 4]
    hat = ha_to_ha_hat  # [6] -> [4, 4]
    exp = lambda ha_: exp_se3(ha_tilde_to_ha(ha_))  # [4, 4] -> [4, 4]
    log = lambda hg_: ha_to_ha_tilde(log_se3(hg_))  # [4, 4] -> [4, 4]
    t = t_se3  # [6] -> [6, 6]
    t_inv = t_inv_se3  # [6] -> [6, 6]


class TestIdentitySE3Identity(TestIdentitiesSE3):
    x = jnp.zeros(6)
