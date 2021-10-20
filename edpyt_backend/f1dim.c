#include "header.h"
#include "util.h"
extern int ncom;
extern double *pcom,*xicom,(*nrfunc)(double []);

double f1dim(double x) {
    int j;
    double f,*xt;
    xt=vector(ncom);
    for (j=0;j<ncom;j++) xt[j]=pcom[j]+x*xicom[j];
    f=(*nrfunc)(xt);
    free_vector(xt,ncom);
    return f;
}
