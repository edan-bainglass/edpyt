#include "util.h"
#include "header.h"
#define TOL 2.0e-4

int ncom;
double *pcom, *xicom, (*nrfunc)(double[]);
void(*nrdfunc)(double[], double[]);

void dlinmin(double p[], double xi[], int n, double *fret, double(*func)(double[]),
	void(*dfunc)(double[], double[]))
{
	int j;
	double xx, xmin, fx, fb, fa, bx, ax;

	ncom = n;
	pcom = vector( n);
	xicom = vector( n);
	nrfunc = func;
	nrdfunc = dfunc;
	for (j = 0; j < n; j++){
		pcom[j] = p[j];
		xicom[j] = xi[j];
	}
	ax = 0.0;
	xx = 1.0;
	mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, f1dim);
	*fret = dbrent(ax, xx, bx, f1dim, df1dim, TOL, &xmin);
	for (j = 0; j < n; j++){
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	free_vector(xicom,n);
	free_vector(pcom,n);
}
