import numpy as np


def stationary_solution(W):
    """Find stationary solution:
    
            dP
            -- = W P =0
            dt
            
    """
    w, v = np.linalg.eig(W)

    j_stationary = np.argmin(abs(w - 1.0))
    p_stationary = v[:,j_stationary].real
    p_stationary /= p_stationary.sum()
    return p_stationary


if __name__=='__main__':
    # /https://core.ac.uk/download/pdf/11546043.pdf
    g12 = 1.
    g21 = 2.
    W = np.array([
        [-g21,g12],
        [g21,-g12]
    ])
    np.testing.assert_allclose([g12/(g12+g21),g21/(g12+g21)],
                               stationary_solution(W))