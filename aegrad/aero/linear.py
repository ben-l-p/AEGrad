from aegrad.aero.uvlm_utils import AeroSnapshot
from aegrad.algebra.base import LinearOperator, BlockLinear
from aegrad.utils import replace_self
from jax import Array
from typing import Sequence, Callable, Optional, Self

input_ordering = ['zeta_b', 'zeta_b_dot']
state_ordering = ['gamma_b', 'gamma_w', 'gamma_bm1', 'zeta_w']
output_ordering = ['f_steady', 'f_unsteady']

class LinearAero:
    def __init__(self, reference: AeroSnapshot):
        self.zeta0_b: Sequence[Array] = reference.zeta_b
        self.zeta0_b_dot: Sequence[Array] = reference.zeta_b_dot
        self.zeta0_w: Sequence[Array] = reference.zeta_w
        self.gamma0_b: Sequence[Array] = reference.gamma_b
        self.gamma0_w: Sequence[Array] = reference.gamma_w
        self.f_steady0: Sequence[Array] = reference.f_steady
        self.f_unsteady0: Sequence[Array] = reference.f_unsteady

        # determine state sizes and slices
        self.n_inputs, self.input_slices = self.make_state_count_slice(self.get_reference_inputs, input_ordering)
        self.n_states, self.state_slices = self.make_state_count_slice(self.get_reference_states, state_ordering)
        self.n_outputs, self.output_slices = self.make_state_count_slice(self.get_reference_outputs, output_ordering)

        # number of states for all vectors
        self.n_all = self.n_inputs | self.n_states | self.n_outputs

        # internal storage for system matrices
        self.sys_entries: dict[tuple[str, str], LinearOperator] = {}
        self._a_mat: Optional[BlockLinear] = None
        self._b_mat: Optional[BlockLinear] = None
        self._c_mat: Optional[BlockLinear] = None
        self._d_mat: Optional[BlockLinear] = None

        self.linearise()

    @property
    def a_mat(self) -> BlockLinear:
        if self._a_mat is None:
            raise ValueError("A matrix has not been constructed.")
        return self._a_mat

    @property
    def b_mat(self) -> BlockLinear:
        if self._b_mat is None:
            raise ValueError("B matrix has not been constructed.")
        return self._b_mat

    @property
    def c_mat(self) -> BlockLinear:
        if self._c_mat is None:
            raise ValueError("C matrix has not been constructed.")
        return self._c_mat

    @property
    def d_mat(self) -> BlockLinear:
        if self._d_mat is None:
            raise ValueError("D matrix has not been constructed.")
        return self._d_mat


    @staticmethod
    def make_state_count_slice(func: Callable[[], dict[str, Sequence[Array]]],
                               ordering: Sequence[str]) -> tuple[dict[str, int], dict[str, slice]]:

        ref_states = func()
        n_states: dict[str, int] = {k: sum([arr.size for arr in ref_states[k]]) for k in ordering}

        state_slices: dict[str, slice] = {}
        cnt = 0
        for state in ordering:
            state_slices.update({state: slice(cnt, cnt + n_states[state])})
            cnt += n_states[state]
        return n_states, state_slices


    def get_reference_inputs(self) -> dict[str, Sequence[Array]]:
        return {'zeta_b': self.zeta0_b, 'zeta_b_dot': self.zeta0_b_dot}

    def get_reference_states(self) -> dict[str, Sequence[Array]]:
        return {
            'gamma_b': self.gamma0_b,
            'gamma_w': self.gamma0_w,
            'gamma_bm1': self.gamma0_b,
            'zeta_w': self.zeta0_w,
        }

    def get_reference_outputs(self) -> dict[str, Sequence[Array]]:
        return {'f_steady': self.f_steady0, 'f_unsteady': self.f_unsteady0}


    def linearise(self) -> None:
        self.make_a_matrix()
        self.make_b_matrix()
        self.make_c_matrix()
        self.make_d_matrix()

    def zero_operator(self, labels: tuple[str, str]) -> dict[tuple[str, str], LinearOperator]:
        return {labels: LinearOperator(None, (self.n_states[labels[0]], self.n_states[labels[1]]))}

    @staticmethod
    def block_matrix_from_dict(entries: dict[tuple[str, str], LinearOperator],
                               input_order: Sequence[str], output_order: Sequence[str]) -> BlockLinear:
        if len(entries) != len(input_order) * len(output_order):
            raise ValueError("Incorrect number of entries provided for BlockLinear construction.")

        blocks = []
        for out_key in output_order:
            blocks.append([])
            for in_key in input_order:
                blocks[-1].append(entries[(out_key, in_key)])
        return BlockLinear(blocks)

    @replace_self
    def make_a_matrix(self) -> Self:
        # corresponds to (d_{output_state}/d_{input_state}) for (output_state, input_state) in state_ordering
        entries: dict[tuple[str, str], LinearOperator] = {}

        # zero entries
        entries.update(self.zero_operator(("gamma_b", "gamma_bm1")))
        entries.update(self.zero_operator(("gamma_w", "gamma_bm1")))
        entries.update(self.zero_operator(("gamma_w", "zeta_w")))
        entries.update(self.zero_operator(("d_gamma_bm1", "gamma_w")))
        entries.update(self.zero_operator(("d_gamma_bm1", "gamma_bm1")))
        entries.update(self.zero_operator(("d_gamma_bm1", "zeta_w")))
        entries.update(self.zero_operator(("zeta_w", "gamma_b")))
        entries.update(self.zero_operator(("zeta_w", "gamma_w")))
        entries.update(self.zero_operator(("zeta_w", "gamma_bm1")))

        # non-zero entries
        entries.update({("gamma_b", "gamma_b"): LinearOperator()})
        entries.update({("gamma_b", "gamma_w"): LinearOperator()})
        entries.update({("gamma_b", "zeta_w"): LinearOperator()})
        entries.update({("gamma_bm1", "gamma_b"): LinearOperator()})    # identity
        entries.update({("gamma_w", "gamma_b"): LinearOperator()})  # convect from TE operator
        entries.update({("gamma_w", "gamma_w"): LinearOperator()})  # shift operator (with consideration of var. wake. disc.
        entries.update({("zeta_w", "zeta_w"): LinearOperator()})    # shift operator (with consideration of var. wake. disc.)

        self._a_mat = self.block_matrix_from_dict(entries, state_ordering, state_ordering)
        self.sys_entries.update(entries)
        return self

    @replace_self
    def make_b_matrix(self) -> Self:
        entries: dict[tuple[str, str], LinearOperator] = {}

        # zero entries
        entries.update(self.zero_operator(("gamma_w", "zeta_b")))
        entries.update(self.zero_operator(("gamma_w", "zeta_b_dot")))
        entries.update(self.zero_operator(("gamma_bm1", "zeta_b")))
        entries.update(self.zero_operator(("gamma_bm1", "zeta_b_dot")))
        entries.update(self.zero_operator(("zeta_w", "zeta_b_dot")))

        # non-zero entries
        entries.update({("gamma_b", "zeta_b"): LinearOperator()})
        entries.update({("gamma_b", "zeta_b_dot"): LinearOperator()})
        entries.update({("zeta_w", "zeta_b"): LinearOperator()})

        self._b_mat = self.block_matrix_from_dict(entries, input_ordering, state_ordering)
        self.sys_entries.update(entries)
        return self

    @replace_self
    def make_c_matrix(self) -> Self:
        entries: dict[tuple[str, str], LinearOperator] = {}

        # zero entries
        entries.update(self.zero_operator(("f_steady", "gamma_bm1")))
        entries.update(self.zero_operator(("f_steady", "zeta_w")))
        entries.update(self.zero_operator(("f_unsteady", "gamma_w")))
        entries.update(self.zero_operator(("f_unsteady", "zeta_w")))

        # non-zero entries
        entries.update({("f_steady", "gamma_b"): LinearOperator()}) # linearised influence and Joukowski forces
        entries.update({("f_steady", "gamma_w"): LinearOperator()}) # linearised influence
        entries.update({("f_unsteady", "gamma_b"): LinearOperator()})  # used to compute gamma_dot
        entries.update({("f_unsteady", "gamma_bm1"): LinearOperator()})  # used to compute gamma_dot

        self._c_mat = self.block_matrix_from_dict(entries, state_ordering, output_ordering)
        self.sys_entries.update(entries)
        return self

    @replace_self
    def make_d_matrix(self) -> Self:
        entries: dict[tuple[str, str], LinearOperator] = {}

        # zero entries
        entries.update(self.zero_operator(("f_unsteady", "zeta_b_dot")))

        # non-zero entries
        entries.update({("f_steady", "zeta_b"): LinearOperator()})    # linearise change in area-normal product
        entries.update({("f_steady", "zeta_b_dot"): LinearOperator()})    # linearise change in area-normal product
        entries.update({("f_unsteady", "zeta_b"): LinearOperator()})    # linearise change in area-normal product

        self._d_mat = self.block_matrix_from_dict(entries, input_ordering, output_ordering)
        self.sys_entries.update(entries)
        return self
