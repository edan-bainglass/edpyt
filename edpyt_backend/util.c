#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"


void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

double *vector(long n)
/* allocate a double vector with subscript range v[0..n] */
{
	double *v;

	v=(double *)malloc((size_t) (n*sizeof(double)));
	if (!v) nrerror("allocation failure in vector()");
	return v;
}


double complex *cvector(long n)
/* allocate a double vector with subscript range v[0..n] */
{
	double complex *v;

	v=(double complex*)malloc((size_t) (n*sizeof(double complex)));
	if (!v) nrerror("allocation failure in vector()");
	return v;
}


double complex **cmatrix(long m, long n)
/* allocate a double vector with subscript range v[0..n] */
{
	double complex **v;

	v = (double complex**)malloc((size_t) (m*sizeof(double complex*)));
    for (int i=0; i<m; i++) {
        v[i] = (double complex*)malloc((size_t) (n*sizeof(double complex)));
	}
	if (!v) nrerror("allocation failure in vector()");
	return v;
}


void free_vector(double *v, long n)
/* free a double vector allocated with vector() */
{
	free((char*) (v));
}


void free_cmatrix(double complex **v, long m)
/* free a double vector allocated with vector() */
{
	for (int i=0; i<m; i++) {
		free(v[i]);
	}
	free(v);
}