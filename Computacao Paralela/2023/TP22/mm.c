/* Sequential time = 
 * Parallel time:  */
/* real    0m9.803s */
/* user    1m15.779s */
/* sys     0m0.037s */
/* Parallel time GPU = */
/* real    0m39.897s */
/* user    0m39.818s */
/* sys     0m0.030s */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mm(double* a, double* b, double* c, int width) 
{
  #pragma omp target teams distribute parallel for map(to: a[0:width*width], b[0:width*width]) map(from: c[0:width*width])
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      double sum = 0;
      for (int k = 0; k < width; k++) {
        double x = a[i * width + k];
        double y = b[k * width + j];
        sum += x * y;
      }
      c[i * width + j] = sum;
    }
  }
}

int main()
{
  int width = 2000;
  double *a = (double*) malloc (width * width * sizeof(double));
  double *b = (double*) malloc (width * width * sizeof(double));
  double *c = (double*) malloc (width * width * sizeof(double));

  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
    }
  }

  #pragma omp target enter data map(to:a[0:width*width],b[0:width*width]) map(alloc:c[0:width*width])
  mm(a,b,c,width);
  #pragma omp target update from(c[0:width*width])
  
   /*  for(int i = 0; i < width; i++) { */
   /*  for(int j = 0; j < width; j++) { */
   /*    printf("\n c[%d][%d] = %f",i,j,c[i*width+j]); */
   /*  } */
   /* } */

  free(a);
  free(b);
  free(c);

  return 0;
}
