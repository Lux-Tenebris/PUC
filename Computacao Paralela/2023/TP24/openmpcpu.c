#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double sum_seq(double *a, int n) {
    double sum = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            local_sum += a[i];
        }
        #pragma omp critical
        sum += local_sum;
    }
    return sum;
}

int main()
{
    omp_set_num_threads(4);
    int width = 40000000;
    double *a = (double*) malloc(sizeof(double) * width);
    printf("Foi\n");
    #pragma omp parallel for
    for(int i = 0; i < width; i++) //Fiz zoado msm
        a[i] = i;
    double s = sum_seq(a, width);
    printf("\nSum = %f\n", s);
}