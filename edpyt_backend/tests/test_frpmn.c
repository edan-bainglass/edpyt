#include <stdio.h>
#include <math.h>
#include "../header.h"

// https://www.wolframalpha.com/input/?i=minimize+2*u%5E2+%2B+3*u*v+%2B+7*v%5E2+%2B+8*u+%2B+9*v+%2B+10

double a = 2., b = 3., c = 7., d = 8., e = 9., f = 10.;


double func(double x[]) {
    return a*pow(x[0],2) + b*x[0]*x[1] + c*pow(x[1],2) + d*x[0] + e*x[1] + f;
}


void dfunc(double x[], double dx[]) {
    dx[0] = 2*a*x[0] + b*x[1] + d;
    dx[1] = b*x[0] + 2*c*x[1] + e;
}


void test_frpmn() {
    int iter = 0, n = 2;
    double p[2] = {0.,0.};
    double sol[2] = {-85./47.,-12./47.};
    double fret = 0., ftol = 1.0e-10;
    frprmn(p, n, ftol, &iter, &fret, func, dfunc);
    CU_ASSERT(sol == p);
}