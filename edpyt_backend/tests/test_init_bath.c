#include <stdio.h>
#include "../bath.h"

int nbath;

int test_init_bath_even(void) {
    printf("Test even\n");
    nbath = 8;
    double* x;
    x = (double*)malloc(2*nbath*sizeof(double));
    init_bath(x);
    for (int j=0; j<2*nbath; j++) {
        printf("%.6f ", x[j]);
    }
    printf("\n");
    return 0;
}

int test_init_bath_odd(void) {
    printf("Test odd\n");
    nbath = 7;
    double* x;
    x = (double*)malloc(2*nbath*sizeof(double));
    init_bath(x);
    for (int j=0; j<2*nbath; j++) {
        printf("%.6f ", x[j]);
    }
    printf("\n");
    return 0;
}

int main(void) {
    test_init_bath_even();
    test_init_bath_odd();
}