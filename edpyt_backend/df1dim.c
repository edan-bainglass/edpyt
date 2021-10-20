#include "util.h"
#include "header.h"
extern int ncom;
extern double *pcom,*xicom,(*nrfunc)(double []);
extern void (*nrdfunc)(double [], double []);

double df1dim(double x) {
    int j;
    double df1=0.0;
    double *xt,*df;
    xt=vector(ncom);
    df=vector(ncom);
    for (j=0;j<ncom;j++) xt[j]=pcom[j]+x*xicom[j];
    (*nrdfunc)(xt,df);
    for (j=0;j<ncom;j++) df1 += df[j]*xicom[j];
    free_vector(df,ncom);
    free_vector(xt,ncom);
    return df1;
}