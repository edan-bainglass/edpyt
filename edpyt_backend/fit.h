#ifndef _FIT_H_
#define _FIT_H_

#include "bath.h"
#include "header.h"

void fit(double x[], int *iter, double *fret, int _nbath, 
         int _nmats, double complex *_vals_true, double _beta);

#endif  /* _FIT_H */