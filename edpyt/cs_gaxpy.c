#include "cs.h"
/* y = A*x+y */

csi csr_gaxpy (const cs *A, const double *x, double *y)
{
  csi p, i, m, *Ap, *Ai ;
  double *Ax ;
  // if (!CS_CSR (A) || !x || !y) return (0) ;       /* check inputs */
  m = A->m ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
  for (i = 0 ; i < m ; i++)
    {
      for (p = Ap [i] ; p < Ap [i+1] ; p++)
        {
    y [i] += Ax [p] * x [Ai [p]] ;
        }
    }
  return (1) ;
}

csi csr_saxpy (const cs *A, const double *x, double *y, csi i, csi n)
{
  csi p, s, *Ap, *Ai ;
  double *Ax ;
  // if (!CS_CSR (A) || !x || !y) return (0) ;       /* check inputs */
  Ap = A->p ; Ai = A->i ; Ax = A->x ;
  for (p = Ap [i] ; p < Ap [i+1] ; p++)
    {
      for (s = 0 ; s < n; s++)
        {
          y[i*n+s] += Ax [p] * x [Ai [p]*n+s] ;
        }
    }
  return (1) ;
}
