#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read
from scipy.interpolate import interp1d

from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc

p = Path('../scatt')

atoms = read(p / 'scatt.xyz')
z = atoms.positions[:, 2]
atoms = atoms[np.where((z > (z.min() + 1)) & (atoms.symbols == 'C'))[0]]

occupancy_goal = np.load(p / 'data_OCCPS.npy')

L = occupancy_goal.size

z_mats = np.load(p / 'data_ENERGIES_MATS.npy')
z_ret = np.load(p / 'data_ENERGIES.npy')
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
gfloc = Gfloc(H - DC, S, HybMats, idx_neq, idx_inv)

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
            max_iter=5,
            tol=1e-1,
            adjust_mu=True,
            alpha=0.)

Sigma = lambda z: np.zeros((nimp, z.size), complex)
delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=0.)

tot_iter = 20

if tot_iter < dmft.max_iter:
    print("tot_iter should be greater than max_iter")

while dmft.it < tot_iter:
    if dmft.it > 0:
        print("Restarting")
    outcome = dmft.solve(delta, verbose=False)
    delta = dmft.delta
    if outcome == "converged":
        print(f"Converged in {dmft.it} steps")
        break
    print(outcome)
    dmft.max_iter += dmft.max_iter

np.save('data_DELTA_DMFT.npy', dmft.delta)
open('mu.txt', 'w').write(str(gfloc.mu))

_Sigma = lambda z: -DC.diagonal()[:, None] - gfloc.mu + gfloc.Sigma(z)[idx_inv]


def save_sigma(sigma_diag):
    L, ne = sigma_diag.shape
    sigma = np.zeros((ne, L, L), complex)

    def save(spin):
        for diag, mat in zip(sigma_diag.T, sigma):
            mat.flat[::(L + 1)] = diag
        np.save('data_SIGMA_DMFT.npy', sigma)

    for spin in range(1):
        save(spin)


save_sigma(_Sigma(z_ret))
