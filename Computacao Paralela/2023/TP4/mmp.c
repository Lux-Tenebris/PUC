#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
Tempos:
Sequencial = 56 segundos
Paralelo = 16 segundos

Validado com diff entre os arquivos de saída
Hardware usado:
  Amd Ryzen 5 - 3500U - 4 Cores 8 Threads
  8GiB Ram com 5,7GiB para o sistema (a apu é gulosa)
*/
 
void mm(double* a, double* b, double* c, int width) 
{
 #pragma omp parallel for shared(a,b,c)
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
 
  omp_set_num_threads(omp_get_num_procs());
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
    }
  }
 
  mm(a,b,c,width);
 
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < width; j++) {
            printf("%0.1f \t",a[i*width+j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < width; j++) {
            printf("%0.1f \t",b[i*width+j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < width; j++) {
            printf("%0.1f \t",c[i*width+j]);
        }
        printf("\n");
    }
}