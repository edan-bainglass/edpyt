#ifndef _CS_H
#define _CS_H
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#define CS_VER 3                    /* CSparse Version */
#define CS_SUBVER 1
#define CS_SUBSUB 2
#define csi int32_t

/* --- primary CSparse routines and data structures ------------------------- */
typedef struct cs_sparse    /* matrix in compressed-column or triplet form */
{
    csi nzmax ;     /* maximum number of entries */
    csi m ;         /* number of rows */
    csi n ;         /* number of columns */
    csi *p ;        /* column pointers (size n+1) or col indices (size nzmax) */
    csi *i ;        /* row indices, size nzmax */
    double *x ;     /* numerical values, size nzmax */
    csi nz ;        /* # of entries in triplet matrix, -1 for compressed-col */
} cs ;

csi csr_gaxpy (const cs *A, const double *x, double *y) ;
csi csr_saxpy (const cs *A, const double *x, double *y, csi offset, csi stride) ;

#define CS_CSC(A) (A && (A->nz == -1))
#define CS_CSR(A) (A && (A->nz == 1))
#endif
