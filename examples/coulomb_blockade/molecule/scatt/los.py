import numpy as np
from ase.io import write
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap

lowdin = True
gpwfile = 'scatt.gpw'
hsfile = 'hs_lolw_k.npy'

atoms, calc = restart(gpwfile, txt=None)
lcao = LCAOwrap(calc)

fermi = calc.get_fermi_level()
H = lcao.get_hamiltonian()
S = lcao.get_overlap()
H -= fermi * S

nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)

z = basis.atoms.positions[:,2]
scatt = np.where(z>(z.min()+(z.max()-z.min())/2))[0]
active = {'C':3}

basis_p = basis[scatt]
index_p = basis_p.get_indices()
index_c = basis_p.extract().take(active)

Usub, eig = subdiagonalize_atoms(basis, H, S, a=scatt)

# Positive projection onto p-z AOs
for idx_lo in index_p[index_c]:
    if Usub[idx_lo-1,idx_lo] < 0.: # change sign
        Usub[:,idx_lo] *= -1

H = rotate_matrix(H, Usub)
S = rotate_matrix(S, Usub)

if lowdin:
    Ulow = lowdin_rotation(H, S, index_p[index_c])

    H = rotate_matrix(H, Ulow)
    S = rotate_matrix(S, Ulow)

    U = Usub.dot(Ulow)

np.save('idx_los.npy', index_p[index_c])
np.save(hsfile, (H[None,...],S[None,...]))
