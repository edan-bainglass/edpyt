#include "fit.h"

int nbath, nmats;
double beta;
double complex* vals_true;
double complex* delta;
double complex* z;
double complex** ddelta;


void fit(double x[], int *iter, double *fret, int _nbath, 
         int _nmats, double complex *_vals_true, double _beta) {
    
    nmats = _nmats;
    nbath = _nbath;
    vals_true = _vals_true;
    beta = _beta;

    int n  = 2 * nbath;
    double ftol = 1e-10;
    z = cvector(nmats);
    delta = cvector(nmats);
    ddelta = cmatrix(nmats, n);
    for (int i=0; i<nmats; i++) {
        z[i] = (2*i+1) * PI/beta *I;
    }

    init_bath(x);
    frprmn(x, n, ftol, iter, fret, eval_chi2, eval_dchi2);

    free(z);
    free(delta);
    free_cmatrix(ddelta, nmats);
}
