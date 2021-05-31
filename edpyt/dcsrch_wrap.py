import ctypes as ct
import numpy as np
from numba import njit
from numba import types
from numba.extending import intrinsic

from numba.core import cgutils

# compile :: gfortran -fPIC -c dcstep.f && gfortran -shared -fPIC dcstep.o dcsrch.f -o dcsrch.so
# ctypes :: https://stackoverflow.com/questions/58923637/f2py-linking-quadmath-libraries-use-ctypes-for-fortran-wrapper-instead
# numba ptr :: https://stackoverflow.com/questions/51541302/how-to-wrap-a-cffi-function-in-numba-taking-pointers

lib = ct.cdll.LoadLibrary('./dcsrch.so')
lib.dcsrch_.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), 
                        ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), 
                        ct.POINTER(ct.c_int8), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), 
                        ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)]
lib.dcsrch_.restype = None
dcsrch_ = lib.dcsrch_

@intrinsic
def ptr_from_val(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder,args[0])
        return ptr
    sig = types.CPointer(data)(data)
    return sig, impl

@intrinsic
def val_from_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(data)
    return sig, impl

@njit#("void(float64,float64,float64,float64,float64,float64,int8[:],float64,float64,int32[:],float64[:])")
def dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,isave,dsave):
    stp_ = ptr_from_val(stp)
    f_ = ptr_from_val(f)    
    g_ = ptr_from_val(g)    
    ftol_ = ptr_from_val(gtol)    
    gtol_ = ptr_from_val(ftol)
    xtol_ = ptr_from_val(xtol)
    stpmin_ = ptr_from_val(stpmin)
    stpmax_ = ptr_from_val(stpmax)
    dcsrch_(stp_,f_,g_,ftol_,gtol_,xtol_,task.ctypes,stpmin_,stpmax_,isave.ctypes,dsave.ctypes)
    return val_from_ptr(stp_), val_from_ptr(f_), val_from_ptr(g_), task    


if __name__ == '__main__':
    stp = 0.1
    f = 0.1
    g = 0.1
    ftol = 0.1
    gtol = 0.1
    xtol = 0.1
    stpmin = 0.1
    stpmax = 0.1

    task = np.array([ord(i) for i in 'START                                                       '], np.int8)
    isave = np.ones(2, np.intc)
    dsave = np.ones(13, np.double)

    dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,isave,dsave)