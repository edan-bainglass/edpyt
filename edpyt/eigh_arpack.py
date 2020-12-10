import numpy as np
from scipy.sparse.linalg.eigen.arpack import _arpack
from scipy.sparse.linalg.eigen.arpack.arpack import (
    DSAUPD_ERRORS,
    DSEUPD_ERRORS,
    _SymmetricArpackParams,
    _ARPACK_LOCK
)
from scipy._lib._util import _aligned_zeros

arpack_int = np.dtype('int32')

tp = 'd'
arpack_solver = _arpack.dsaupd
arpack_extract = _arpack.dseupd

# def eigsh(n, nev, matvec, v0=None):
#     params = _SymmetricArpackParams(n, nev, tp, matvec,
#                                     which='SA', v0=v0)
#     with _ARPACK_LOCK:
#         while not params.converged:
#             params.iterate()
#
#         return params.extract(return_eigenvectors=True)

#    %------------------------------------------------------%
#    | Storage Declarations:                                |
#    |                                                      |
#    | The maximum dimensions for all arrays are            |
#    | set here to accommodate a problem size of            |
#    | N .le. MAXN                                          |
#    |                                                      |
#    | NEV is the number of eigenvalues requested.          |
#    |     See specifications for ARPACK usage below.       |
#    |                                                      |
#    | NCV is the largest number of basis vectors that will |
#    |     be used in the Implicitly Restarted Arnoldi      |
#    |     Process.  Work per major iteration is            |
#    |     proportional to N*NCV*NCV.                       |
#    |                                                      |
#    | You must set:                                        |
#    |                                                      |
#    | MAXN:   Maximum dimension of the A allowed.          |
#    | MAXNEV: Maximum NEV allowed.                         |
#    | MAXNCV: Maximum NCV allowed.                         |
#    %------------------------------------------------------%

class ArpackParams:
    def __init__(self, maxn, maxncv, ldv, v0=None):

        #    %--------------%
        #    | Local Arrays |
        #    %--------------%

        self.v = np.zeros((ldv,maxncv), tp)
        self.workl = _aligned_zeros(maxncv*(maxncv+8), tp)
        self.workd = _aligned_zeros(3*maxn, tp)
        self.resid = np.zeros(maxn, tp)
        if v0 is None:
            self.info = 0
        else:
            self.resid[:v0.size] = v0
            self.info = 1
        self.select = np.zeros(maxncv, 'int')

        self.iparam = np.zeros(11, arpack_int)
        self.ipntr = np.zeros(11, arpack_int)


def eigsh(n, nev, matvec, v0=None, arpack_param=None):

    #    %-------------------------------------------------%
    #    | The following sets dimensions for this problem. |
    #    %-------------------------------------------------%

    ncv   = min(max(2*nev+1, 20), n)
    bmat  = 'I'
    which = 'SA'
    sigma = 0.

    if arpack_param is None:
        ldv = n
        arpack_param = ArpackParams(n, ncv, ldv, v0)
        # arpack_param = _SymmetricArpackParams(n, ncv, tp, matvec)

    v = arpack_param.v
    workl = arpack_param.workl
    workd = arpack_param.workd
    resid = arpack_param.resid
    info = arpack_param.info
    select = arpack_param.select
    iparam = arpack_param.iparam
    ipntr = arpack_param.ipntr
    maxn = arpack_param.resid.size
    maxncv = arpack_param.v.shape[1]

    assert n<=maxn, 'n is greater than maxn'
    assert ncv<=maxncv, 'ncv is greater than maxncv'

    #    %-----------------------------------------------------%
    #    |                                                     |
    #    | Specification of stopping rules and initial         |
    #    | conditions before calling DSAUPD                    |
    #    |                                                     |
    #    | TOL  determines the stopping criterion.             |
    #    |                                                     |
    #    |      Expect                                         |
    #    |           abs(lambdaC - lambdaT) < TOL*abs(lambdaC) |
    #    |               computed   true                       |
    #    |                                                     |
    #    |      If TOL .le. 0,  then TOL <- macheps            |
    #    |           (machine precision) is used.              |
    #    |                                                     |
    #    | IDO  is the REVERSE COMMUNICATION parameter         |
    #    |      used to specify actions to be taken on return  |
    #    |      from DSAUPD. (See usage below.)                |
    #    |                                                     |
    #    |      It MUST initially be set to 0 before the first |
    #    |      call to DSAUPD.                                |
    #    |                                                     |
    #    | INFO on entry specifies starting vector information |
    #    |      and on return indicates error codes            |
    #    |                                                     |
    #    |      Initially, setting INFO=0 indicates that a     |
    #    |      random starting vector is requested to         |
    #    |      start the ARNOLDI iteration.  Setting INFO to  |
    #    |      a nonzero value on the initial call is used    |
    #    |      if you want to specify your own starting       |
    #    |      vector (This vector must be placed in RESID.)  |
    #    |                                                     |
    #    | The work array WORKL is used in DSAUPD as           |
    #    | workspace.  Its dimension LWORKL is set as          |
    #    | illustrated below.                                  |
    #    |                                                     |
    #    %-----------------------------------------------------%

    lworkl = ncv*(ncv+8)
    tol = 0.
    ido = 0
    ierr = 0

    ishfts = 1
    maxitr = n*10
    mode = 1
    iparam[0] = ishfts
    iparam[2] = maxitr
    iparam[3] = 1
    iparam[6] = mode

    while True:

    #       %---------------------------------------------%
    #       | Repeatedly call the routine DSAUPD and take |
    #       | actions indicated by parameter IDO until    |
    #       | either convergence is indicated or maxitr   |
    #       | has been exceeded.                          |
    #       %---------------------------------------------%

        ido, tol, resid, v, iparam, ipntr, info = \
            arpack_solver(ido, bmat, which, nev, tol, resid,
                          v, iparam, ipntr, workd, workl,
                          info)#, n=n, ncv=ncv, ldv=ldv, lworkl=lworkl)

    #          %--------------------------------------%
    #          | Perform matrix vector multiplication |
    #          |              y <--- OP*x             |
    #          | The user should supply his/her own   |
    #          | matrix vector multiplication routine |
    #          | here that takes workd(ipntr(1)) as   |
    #          | the input, and return the result to  |
    #          | workd(ipntr(2)).                     |
    #          %--------------------------------------%

        xslice = slice(ipntr[0] - 1, ipntr[0] - 1 + n)
        yslice = slice(ipntr[1] - 1, ipntr[1] - 1 + n)

        if (ido == 1) or (ido == -1):
            workd[yslice] = matvec(workd[xslice])

        else:
            break

    if info < 0:
        raise RuntimeError(f'Error with dsaupd, info = {DSAUPD_ERRORS[info]}')

    #       %-------------------------------------------%
    #       | No fatal errors occurred.                 |
    #       | Post-Process using DSEUPD.                |
    #       |                                           |
    #       | Computed eigenvalues may be extracted.    |
    #       |                                           |
    #       | Eigenvectors may be also computed now if  |
    #       | desired.  (indicated by rvec = .true.)    |
    #       |                                           |
    #       | The routine DSEUPD now called to do this  |
    #       | post processing (Other modes may require  |
    #       | more complicated post processing than     |
    #       | mode1.)                                   |
    #       |                                           |
    #       %-------------------------------------------%

    rvec = True

    d, z, ierr = arpack_extract(rvec, 'A', select, sigma,
                   bmat, which, nev, tol, resid, v,
                   iparam[:7], ipntr, workd[:2*n], workl, ierr)#,
                   # n=n, ncv=ncv, ldv=ldv, lworkl=lworkl)

    if ierr != 0:
        raise RuntimeError(f'Error with dseupd, info = {DSEUPD_ERRORS[info]}')

    nconv = iparam[4]
    d = d[:nconv]
    z = z[:, :nconv]

    return d, z
