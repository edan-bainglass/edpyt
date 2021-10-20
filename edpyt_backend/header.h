#ifndef _NR_H_
#define _NR_H_

double f1dim(double x);
double df1dim(double x);
void dlinmin(double p[], double xi[], int n, double *fret, double(*func)(double[]),
	void(*dfunc)(double[], double[]));
double dbrent(double ax, double bx, double cx, double(*f)(double),
	double(*df)(double), double tol, double *xmin);    
void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc,
	double(*func)(double));
void frprmn(double p[], int n, double ftol, int *iter, double *fret,
            double (*func)(double []), void (*dfunc)(double [], double []));

#endif