import sys
import re
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from ase.io import read

from qtpyt.tools import remove_pbc  #, rotate_couplings
from qtpyt.block_tridiag import graph_partition, greenfunction
#from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.basis import Basis
from qtpyt.projector import expand
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
#from qtpyt.continued_fraction import get_ao_charge
#from qtpyt.projector import ProjectedGreenFunction
#from qtpyt.hybridization import Hybridization

pl_path = Path('../leads/')
cc_path = Path('../scatt/')

h_pl_k, s_pl_k = np.load(pl_path / 'hs_leads_k.npy')
h_cc_k, s_cc_k = map(lambda m: m.astype(complex),
                     np.load(cc_path / f'hs_lolw_k.npy'))

basis = {'C': 9, 'H': 4}

atoms_pl = read(pl_path / 'leads.xyz')
basis_pl = Basis.from_dictionary(atoms_pl, basis)

atoms_cc = read(cc_path / 'scatt.xyz')
basis_cc = Basis.from_dictionary(atoms_cc, basis)

h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(
    lambda m: m[0],
    prepare_leads_matrices(h_pl_k,
                           s_pl_k, (3, 1, 1),
                           align=(0, h_cc_k[0, 0, 0]))[1:])
del h_pl_k, s_pl_k
remove_pbc(basis_cc, h_cc_k)
remove_pbc(basis_cc, s_cc_k)

se = [None, None, None]
se[0] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij))
se[1] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id='right')

#rotate_couplings(basis_pl, se[0], Nr)
#rotate_couplings(basis_pl, se[1], Nr)

nodes = [0, basis_pl.nao, basis_cc.nao - basis_pl.nao, basis_cc.nao]

hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, h_cc_k[0], s_cc_k[0])
del h_cc_k, s_cc_k

de = 0.01
energies = np.arange(-3, 3 + de / 2., de).round(7)
eta = 1e-4  #2*de

gf = greenfunction.GreenFunction(hs_list_ii,
                                 hs_list_ij, [(0, se[0]),
                                              (len(hs_list_ii) - 1, se[1])],
                                 solver='dyson',
                                 eta=eta)
i1 = np.load(cc_path / 'idx_los.npy') - nodes[1]
s1 = hs_list_ii[1][1]


class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(s1, super().retarded(energy), i1)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


def run():

    global suffix

    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(f'data_TRANSMISSION_{suffix}.npy', T.real)


# DFT
suffix = 'DFT'
run()
for fn in Path('../dmft/').glob('data_SIGMA_DMFT_DMU*'):
    se[2] = load(fn)
    suffix = 'DMFT_DMU_' + list(re.findall(r'-?\d+\.\d+', str(fn)))[0]
    gf.selfenergies.append((1, se[2]))
    run()
    gf.selfenergies.pop()
