import numpy as np
from ase.io import read
from gpaw import GPAW, FermiDirac, Mixer
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank

atoms = read('leads.xyz')

calc = GPAW(h=0.2,
            xc='LDA',
            nbands='nao',
            convergence={'bands':'all'},
            basis='szp(dzp)',
            occupations=FermiDirac(width=0.01),
            kpts=(3, 1, 1),
            mode='lcao',
            txt='leads.txt',
            mixer=Mixer(0.02, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': True})

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('leads.gpw')

fermi = calc.get_fermi_level()
print(repr(fermi), file=open('fermi_leads.txt', 'w'))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= calc.get_fermi_level() * S_kMM
    np.save('hs_leads_k.npy', (H_kMM, S_kMM))
