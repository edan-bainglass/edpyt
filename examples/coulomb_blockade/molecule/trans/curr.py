from os import walk
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from ase.units import _e, _hplanck, kB

G0 = 2. * _e**2 / _hplanck


def fermidistribution(energy, kt):
    # fermi level is fixed to zero
    # energy can be a single number or a list
    assert kt >= 0., 'Negative temperature encountered!'

    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2. <= 0)
        else:
            return (energy / 2. <= 0).astype(int)
    else:
        return 1. / (1. + np.exp(energy / kt))


def current(bias, E, T_e, T=300, unit='uA'):
    """Get the current in nA."""
    if not isinstance(bias, (int, float)):
        bias = bias[np.newaxis]
        E = E[:, np.newaxis]
        T_e = T_e[:, np.newaxis]

    fl = fermidistribution(E - bias / 2., kB * T)
    fr = fermidistribution(E + bias / 2., kB * T)

    return G0 * np.trapz((fl - fr) * T_e, x=E, axis=0) * 1e6  # uA


def numerical_derivative(x, y):

    dy_dx = np.diff(y) / np.diff(x)
    dy_dx = np.append(dy_dx, np.nan)

    return dy_dx


E = np.load('../scatt/data_ENERGIES.npy').real

files = []
directory = 'transmission_data/'
for (dirpath, dirnames, filenames) in walk(directory):
    files.extend(f'{directory}{file}' for file in filenames)
files.sort(key=lambda fn: float(fn.split('_')[-1].strip(".npy")))

dV = 0.1
Vmin = -1.5
Vmax = 1.5 + dV / 2.
dmu_min = -4
dmu_max = 3
bias = np.arange(-2.5, 2.5 + dV / 2., dV)
T = np.asarray([np.load(fn) for fn in files])
I = np.asarray([current(bias, E, t, T=9) for t in T])
# I = np.clip(I, 0, 30)

dI_dV = np.asarray([numerical_derivative(bias, i) for i in I])
log_dI_dV = np.log(dI_dV)
plt.imshow(log_dI_dV.T,
           interpolation='sinc',
           origin='lower',
           extent=(dmu_min, dmu_max, Vmin, Vmax))
plt.xlabel('Gate Voltage V$_G$ (V)')
plt.ylabel('Bias V$_{DS}$ (V)')
plt.colorbar()
plt.savefig('blockade.png', bbox_inches='tight', dpi=300)
# plt.show()
