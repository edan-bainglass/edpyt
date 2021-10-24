#include <stdio.h>
#include "bath.h"
#include "util.h"

extern int nbath, nmats;
// extern double beta;
extern double complex* vals_true;
extern double complex* delta;
extern double complex* z;
extern double complex** ddelta;


extern void dgemv_(char* trans, int* m, int* n, double* alpha, double* A,
            int* lda, double* x, int* incx, double* beta, double* y, int* incy);


double BANDWIDTH = 2.0;


void init_bath(double x[]) {
    int j;
    double* ek = &x[0];
    double* vk = &x[nbath];
    int nhalf = nbath / 2;
    double de;
    // Init hoppings.
    for (j=0; j<nbath; j++) {
        vk[j] = MAX(0.1, 1/sqrt((double)nbath));
    }
    ek[0] = -BANDWIDTH;
    ek[nbath-1] = BANDWIDTH;
    // even :: [-2,-1,-0.1,0.1,+1,+2]
    if ((!(nbath&1))&&(nbath>=4)) {
        de = BANDWIDTH/(double)MAX(nhalf-1,1);
        ek[nhalf-1] = -0.1;    
        ek[nhalf] = 0.1;    
        for (j=1; j<nhalf-1; j++) {
            ek[j] = -BANDWIDTH + j*de;
            ek[nbath-j-1] = BANDWIDTH - j*de;
        }
    }
    // odd :: [-2,-1,0,+1,+2]
    else if ((nbath&1)&&(nbath>3)) {
        de = BANDWIDTH/(double)nhalf;
        ek[nhalf] = 0.;
        for (j=1; j<nhalf; j++) {
            ek[j] = -BANDWIDTH + j*de;
            ek[nbath-j-1] = BANDWIDTH - j*de;
        }
    }
}


void eval_delta(double x[]) {
    int i, j;
    double* ek = x;
    double* vk = x + nbath;
    double* vk2 = vector(nbath);
    for (j=0; j<nbath; j++) {
        vk2[j] = pow(vk[j], 2);
    }
    for (i=0; i<nmats; i++) {
        delta[i] = 0.;
        for (j=0; j<nbath; j++) {
            delta[i] += vk2[j] / (z[i] - ek[j]);
        }
    }
    free(vk2);
}


void eval_ddelta(double x[]) {
    int i, j;
    double* ek = x;
    double* vk = x + nbath;
    double* vk2 = vector(nbath);
    double complex denom;
    for (j=0; j<nbath; j++) {
        vk2[j] = pow(vk[j], 2);
    }
    for (i=0; i<nmats; i++) {
        for (j=0; j<nbath; j++) {
            ddelta[i][j] = vk2[j] / cpow(z[i]-ek[j], 2);
            ddelta[i][j+nbath] = 2. * vk[j] / (z[i]-ek[j]);
        }
    }
    free(vk2);
}


double eval_chi2(double x[]) {
    double chi2 = 0.;
    eval_delta(x);
    for (int i=0; i<nmats; i++) {
        chi2 += pow(cabs(vals_true[i]-delta[i]), 2);
    }
    chi2 /= (double)nmats;
    return chi2;
}


void eval_dchi2(double x[], double dx[]) {
    double complex F;
    int n = 2*nbath;
    int i, k;
    for (int k=0; k<n; k++) {
        dx[k] = 0.;
    }
    eval_delta(x);
    eval_ddelta(x);
    for (int i=0; i<nmats; i++) {
        F = vals_true[i]-delta[i];
        for (int k=0; k<n; k++) {
            dx[k] += creal(ddelta[i][k]) * creal(F) + cimag(ddelta[i][k]) * cimag(F);
        }
    }
    for (int k=0; k<n; k++) {
        dx[k] = -2.*dx[k]/(double)nmats;
    }
}
