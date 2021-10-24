#include <stdio.h>
#include "../bath.h"
#include "../util.h"

int nmats = 3000, nbath = 8;
double beta = 70.;
double complex* z;
double complex* delta;
double complex** ddelta;

int main(void) {
    FILE* fp_delta;
    FILE* fp_ddelta;
    fp_delta = fopen("delta.txt","w");
    fp_ddelta = fopen("ddelta.txt","w");
    int n = 2*nbath;
    double x[] = {-1.98663495, -1.28445211, -0.42108352, -0.05721278,  0.05721278,  0.42108352,
                  1.28445211,  1.98663495,  0.08997913,  0.18150846,  0.59537434,  0.31923968,
                  0.31923968,  0.59537434,  0.18150846,  0.08997913};
    z = cvector(nmats);
    for (int i=0; i<nmats; i++) {
        z[i] = (2*i+1) * PI/beta *I;
    }
    delta = cvector(nmats);
    ddelta = cmatrix(nmats,n);
    eval_delta(x);
    eval_ddelta(x);
    for (int i=0; i<nmats; i++) {
        fprintf(fp_delta, "%.6f %.6f\n", creal(delta[i]), cimag(delta[i]));
        for (int j=0; j<n; j++) {
            fprintf(fp_ddelta, "%.6f %.6f ", creal(ddelta[i][j]), cimag(ddelta[i][j]));
        }
        fprintf(fp_ddelta, "\n");
    }
    fclose(fp_delta);
    fclose(fp_ddelta);
}