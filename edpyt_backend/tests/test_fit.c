#include "../fit.h"
#include <stdio.h>
#include <time.h>

void hybrid_true(double complex vals_true[], int nmats, double beta) {
    double complex zi;
    for (int i=0; i<nmats; i++) {
        zi = (2*i+1) * PI/beta * I;
        vals_true[i] = 2*(zi-csqrt(cpow(zi,2.)-1.));
    }
}


void test_hybrid(void) {
    int nmats = 3000, nbath = 8, iter=0;
    double beta = 70., fret;
    double complex* vals_true = (double complex*)malloc(nmats*sizeof(double complex));
    clock_t start;
    double cpu_time_used;
    double* x = (double*)malloc(2*nbath*sizeof(double));
    hybrid_true(vals_true, nmats, beta);
    start = clock();
    fit(x, &iter, &fret, nbath, nmats, vals_true, beta);
    cpu_time_used = ((double) (clock() - start)) / CLOCKS_PER_SEC;
    printf("Fit took %.6f seconds.\n", cpu_time_used);
    printf("Fit completed with %d iterations.\n", iter);
    printf("Current function value is %.6f.\n", fret);
    for (int j=0; j<2*nbath; j++) {
        printf("%.6f ", x[j]);
    }
    printf("\n");
}


int main(int argc, char* argv[]) {
    test_hybrid();
}