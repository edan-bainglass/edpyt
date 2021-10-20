#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#define FREE_ARG char*

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

void free_vector(double *v, long n)
/* free a double vector allocated with vector() */
{
	free((FREE_ARG) (v));
}