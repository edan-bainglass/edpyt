import numpy as np
from pathlib import Path

dir = Path('.test')

for file in dir.glob('*.npy'):
    expected = np.load(file)
    computed = np.load(file.stem+'.npy')
    np.testing.assert_allclose(expected, computed)