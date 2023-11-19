#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read
from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc
from edpyt.pprint import pprint
from scipy.interpolate import interp1d

p = Path('../scatt')

atoms = read(p / 'scatt.xyz')
z = atoms.positions[:, 2]
atoms = atoms[np.where((z > (z.min() + 1)) & (atoms.symbols == 'C'))[0]]

occupancy_goal = np.load(p / 'data_OCCPS.npy')

L = occupancy_goal.size

z_mats = np.load(p / 'data_ENERGIES_MATS.npy')

beta = np.pi / (z_mats[0].imag)

hyb_mats = np.fromfile(
    p / 'data_HYBRID_MATS.bin',
    complex,
).reshape(z_mats.size, L, L)

_HybMats = interp1d(z_mats.imag,
                    hyb_mats,
                    axis=0,
                    bounds_error=False,
                    fill_value=0.)
HybMats = lambda z: _HybMats(z.imag)

H = np.load(p / 'data_HAMILTON.npy').real
S = np.eye(L)

idx_neq = np.arange(L)
idx_inv = np.arange(L)

U = 4.  # Interaction
V = np.eye(L) * U
DC = np.diag(V.diagonal() * (occupancy_goal - 0.5))
gfloc = Gfloc(H - DC, np.eye(L), HybMats, idx_neq, idx_inv)

nimp = gfloc.idx_neq.size
gfimp: list[Gfimp] = []
n = 4

for i in range(nimp):
    gfimp.append(Gfimp(n, z_mats.size, V[i, i], beta))

gfimp = nanoGfimp(gfimp)

occupancy_goal = occupancy_goal[gfloc.idx_neq]
dmft = DMFT(gfimp,
            gfloc,
            occupancy_goal,
            max_iter=1,
            tol=9e-1,
            adjust_mu=True,
            alpha=0.)

Sigma = lambda z: np.zeros((nimp, z.size), complex)
delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=0.)

tot_iter = 1

while dmft.it < tot_iter:
    if dmft.it > 0:
        pprint("Restarting")
    dmft.solve(delta, verbose=False)
    dmft.max_iter += dmft.max_iter
