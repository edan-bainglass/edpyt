#ifndef _BATH_H_
#define _BATH_H_

#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <math.h>
#define PI (double)3.14159265358979323846
#define MAX(A, B) (A > B ? A : B)

void init_bath(double x[]);
double delta_chi2(double x[]);
void delta_dchi2(double x[], double dfx[]);

#endif /* _BATH_H */