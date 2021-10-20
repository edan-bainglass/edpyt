#include "bath.h"

extern int nbath, nmats;
extern double beta;
extern double complex* vals_true;

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

double delta_chi2(double x[]) {
    int i, j;
    double complex zi, val;    
    double cdst;
    // Return value
    double fx = 0.;
    for (i=0; i<nmats; i++) {
        val = 0.;
        zi = (2*i+1) * PI/beta * I;
        for (j=0; j<nbath; j++) {
            val += pow(x[j+nbath],2) / (zi - x[j]);
        }
        cdst = cabs(vals_true[i]-val); //complex distance
        fx += pow(cdst, 2);
    }
    fx /= nmats;
    return fx;
}


void delta_dchi2(double x[], double dx[]) {
    int i, j, n = 2*nbath;
    double complex zi, val, denom, dst;    
    double complex *dfx;
    dfx = (double  complex*)malloc((size_t) (n*sizeof(double complex)));    
    // Return value
    for (j=0; j<n; j++) {
        dx[j] = 0.;
    }
    for (i=0; i<nmats; i++) {
        val = 0.;
        zi = 2*(i+1) * PI/beta * I;
        for (j=0; j<nbath; j++) {
            denom = (zi - x[j]);
            val += pow(x[j+nbath],2) / denom;
            dfx[j] = pow(x[j+nbath],2) / pow(denom, 2); 
            dfx[j+nbath] = 2*x[j+nbath] / denom; 
        }
        dst = vals_true[i]-val; // complex distance
        for (j=0; j<n; j++) {
            dx[j] += creal(dfx[j]) * creal(dst) + cimag(dfx[j]) * cimag(dst);
        }
    }
    for (j=0; j<n; j++) {
        dx[j] *= -2./nmats;
    }
}