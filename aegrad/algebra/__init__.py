r"""
Naming conventions:
ha, d: Lie algebra vector for SE(3), [6]
ha_u, ha_omega: Lie algebra vector linear and rotational components, [3], [3]
ha_tilde: Lie algebra element, [4, 4]
ha_hat: Adjoint representation, [6, 6]
hg: Lie group element for SE(3), [4, 4]
x: Node coordinate, [3]
rmat: Rotation matrix, [3, 3]
omega: Rotation vector, [3]
vec: General vector, [...]
_matrix: General matrix, [..., ...]
skew: Skew symmetric matrix, [3, 3]
log_: Logarithm operator, SE|SO -> se|so
exp_: Exponential operator, se|so -> SE|SO
t_: Tangent operator
bracket_: Lie bracket operator
s_l: Relative beam-wise coordinate
_full, _small_ang, _zero: Internal names for full function, small angle approximation and zero value function
"""