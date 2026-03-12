# def _d_v_mp_zeta_b(
#     zeta_b_to_x: Literal["chordwise_mp", "spanwise_mp", "direct"],
#     gamma_b: ArrayList,
#     gamma_w: ArrayList,
#     zeta_b: ArrayList,
#     zeta_w: ArrayList,
#     kernels: Sequence[KernelFunction],
#     flowfield: FlowField,
#     mirror_point: Optional[Array],
#     mirror_normal: Optional[Array],
#     t: Array,
# ) -> Array:
#     r"""
#     :param gamma_b:
#     :param gamma_w:
#     :param zeta_b:
#     :param zeta_w:
#     :param kernels:
#     :param flowfield:
#     :param mirror_point:
#     :param mirror_normal:
#     :param t:
#     :return:
#     """
#
#     def func_zeta_b_to_x(zeta_b_: Array) -> Array:
#         match zeta_b_to_x:
#             case "chordwise_mp":
#                 return 0.5 * (zeta_b_[:-1, :, :] + zeta_b_[1, :, :])
#             case "spanwise_mp":
#                 return 0.5 * (zeta_b_[:-1, :, :] + zeta_b_[1, :, :])
#             case "direct":
#                 return zeta_b_
#             case _:
#                 raise ValueError(f"Invalid value for zeta_b_to_x: {zeta_b_to_x}")
#
#     x_target = (
#         ArrayList([func_zeta_b_to_x(zeta_b_) for zeta_b_ in zeta_b])
#         .flatten()
#         .reshape(-1, 3)
#     )  # [n_x, 3]
#
#     def get_v_inf_grad(x_target: Array) -> Array:
#         r"""
#         :param x_target: Coordinate, [3]
#         :return: Gradient of the freestream velocity with respect to the coordinate, [3, 3]
#         """
#         return jacrev(flowfield, argnums=0)(x_target, t)
#
#     d_v_inf_d_x = vmap(get_v_inf_grad, 0, 0)(x_target)  # [n_x, 3, 3]
#
#
# def _d_f_steady_d_zeta_b():
#     pass
