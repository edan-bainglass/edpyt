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
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.basis import Basis
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.continued_fraction import get_ao_charge
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.hybridization import Hybridization
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from scipy.linalg import eigvalsh

pl_path = Path('../leads/')
cc_path = Path('./')

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

se = [None, None]
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

ibf_los = np.load(cc_path / 'idx_los.npy')
gfp = ProjectedGreenFunction(gf, ibf_los)
hyb = Hybridization(gfp)

no = len(ibf_los)
gd = GridDesc(energies, no, complex)
HB = gd.empty_aligned_orbs()
D = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    HB[e] = hyb.retarded(energy)
    D[e] = gfp.get_dos(energy)

if comm.rank == 0:
    for fn in cc_path.glob('*.bin'):
        Path.unlink(fn)
D = gd.gather_energies(D)
gd.write(HB, 'data_HYBRID.bin')

if comm.rank == 0:
    ne = energies.size
    np.save('data_PDOS.npy', D.real)
    np.save('data_ENERGIES.npy', energies + 1.j * eta)

if comm.rank == 0:
    Heff = (hyb.H + hyb.retarded(0.)).real
    np.save('data_HAMILTON.npy', hyb.H)
    np.save('data_HAMILTON_EFF.npy', Heff)
    np.save('data_EIGVALS.npy', eigvalsh(Heff, gfp.S))

# Matsubara
gf.eta = 0.
assert se[0].eta == 0.
assert se[1].eta == 0.
ne = 3000
beta = 70.
energies = 1.j * (2 * np.arange(ne) + 1) * np.pi / beta
gd = GridDesc(energies, no, complex)
HB = gd.empty_aligned_orbs()

for e, energy in enumerate(gd.energies):
    HB[e] = hyb.retarded(energy)

gd.write(HB, 'data_HYBRID_MATS.bin')

if comm.rank == 0:
    np.save('data_OCCPS.npy', get_ao_charge(gfp))
    np.save('data_ENERGIES_MATS.npy', energies)
