#include <stdio.h>
#include <math.h>
#include "header.h"
#include "util.h"
#define ITMAX 200
#define EPS 1.0e-10
#define FREEALL free_vector(xi,n);free_vector(h,n);free_vector(g,n);


void frprmn(double p[], int n, double ftol, int *iter, double *fret,
            double (*func)(double []), void (*dfunc)(double [], double [])) {

    /*
    Given a starting point p[1..n], Fletcher-Reeves-Polak-Ribiere 
    minimization is performed on afunction func, using its gradient 
    as calculated by a routinedfunc.  The convergence toleranceon 
    the function value is input asftol.  Returned quantities are :
        p : the location of the minimum,
        iter : the  number  of  iterations  that  were  performed,
        fret : the  minimum  value  of  thefunction.
    */
    int j,its;
    double gg,gam,fp,dgg;
    double *g,*h,*xi;
    g=vector(n);
    h=vector(n);
    xi=vector(n);
    fp=(*func)(p);
    (*dfunc)(p,xi);
    for (j=0;j<n;j++) {
        g[j] = -xi[j];
        xi[j]=h[j]=g[j];
    }
    ftol=1.0e-10;
    for (its=1;its<=ITMAX;its++) {
        *iter=its;
        dlinmin(p,xi,n,fret,func,dfunc);
        if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
            FREEALL
            return;
        }
        fp= *fret;
        (*dfunc)(p,xi);
        dgg=gg=0.0;
        for (j=0;j<n;j++) {
            gg += g[j]*g[j];
            dgg += (xi[j]+g[j])*xi[j];
        }
        if (gg == 0.0) {
            FREEALL
            return;
        }
        gam=dgg/gg;
        for (j=0;j<n;j++) {
            g[j] = -xi[j];
            xi[j]=h[j]=g[j]+gam*h[j];
        }
    }
    nrerror("Too many iterations in frprmn");
}