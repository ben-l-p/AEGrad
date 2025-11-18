from __future__ import annotations
from aegrad.aero.case import AeroCase
from aegrad.array_utils import neighbour_average

def calculate_f_steady(case: AeroCase, i_ts: int) -> AeroCase:
    for i_surf in range(case.n_surf):
        mp_chordwise = neighbour_average(case.zeta_b[i_surf][i_ts, ...], axes=0) # [zeta_m-1, zeta_n, 3]
        mp_spanwise = neighbour_average(case.zeta_b[i_surf][i_ts, ...], axes=1) # [zeta_m, zeta_n-1, 3]


def calculate_f_unsteady(case: AeroCase, i_ts: int) -> AeroCase:
    pass