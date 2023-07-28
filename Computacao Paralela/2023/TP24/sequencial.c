#include <stdio.h>
#include <stdlib.h>

double sum_seq(double* a, int n)
{
  double sum = 0.0;
  for (int i = 0; i < n; i++)
  {
    sum += a[i];
  }
  return sum;
}

int main()
{
    int width = 40000000;
    double *a = (double*) malloc(sizeof(double) * width);
    printf("Foi\n");
    for(int i = 0; i < width; i++) //Fiz zoado msm
        a[i] = i;
    double s = sum_seq(a, width);
    printf("\nSum = %f\n", s);
}