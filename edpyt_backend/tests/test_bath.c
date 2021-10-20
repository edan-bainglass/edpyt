#include <stdio.h>
#include "../bath.h"

int nbath=2, nmats = 5;
double complex vals_true[] = {1.,1.,1.,1.,1.};
double beta = 1.;


void print(double *res, double *expect, int n) {
    for (int j=0; j<n; j++) {
        printf("Result : %.6", res[j]);
        printf("Result : %.6", expect[j]);

    }
}


void test_delta_chi2() {
    double x[] = {1.,1.,1.,1.};
    double chi2 = delta_chi2(x);
    double expect = 1.176770503334895;
    print(&chi2, &expect, 1);
}


void test_delta_dchi2() {
    double x[] = {1.,1.,1.,1.};
    double dchi2[] = {0.,0.,0.,0.};
    delta_dchi2(x, dchi2);
    print(dchi2, dchi2, 4);
}


int main(void) {
    test_delta_chi2();
    test_delta_dchi2();
}

/*
z = 1.j*(2*np.arange(5)+1)*np.pi/1.
delta = (np.ones(2)[None,:] / (z[:,None]-np.ones(2)[None,:])).sum(1)
chi2 = (abs(1-delta)**2).sum()/5
ddelta = np.empty(4,complex)
ddelta[:2] = (np.ones(2)[None,:] / (z[:,None]-np.ones(2)[None,:])**2).sum(1)
ddelta[2:] = (2* np.ones(2)[None,:] / (z[:,None]-np.ones(2)[None,:])).sum(1)

*/