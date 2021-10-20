#include "fit.h"


int nbath, nmats;
double complex* vals_true;
double beta;


void fit(double x[], int *iter, double *fret, int _nbath, 
         int _nmats, double complex *_vals_true, double _beta) {
    
    nmats = _nmats;
    nbath = _nbath;
    vals_true = _vals_true;
    beta = _beta;

    int n  = 2 * nbath;
    double ftol = 1e-10;

    init_bath(x);
    frprmn(x, n, ftol, iter, fret, delta_chi2, delta_dchi2);

}
