#include <stdio.h>
#include <stdlib.h>

__device__ double atomicaAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void sum_cuda(double* a, double* s, int width) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int sum = 0;

  __syncthreads();

  while (tid < width) {
    sum += a[tid];
    tid += blockDim.x * gridDim.x;
  }

  __syncthreads();

  atomicaAdd(s, sum);
}


int main()
{
  int width = 40000000;
  int size = width * sizeof(double);

  int block_size = 1024;
  int num_blocks = (width-1)/block_size+1;
  int s_size = (num_blocks * sizeof(double));  
 
  double *a = (double*) malloc (size);
  double *s = (double*) malloc (s_size);

  for(int i = 0; i < width; i++)
    a[i] = i;

  double *d_a, *d_s;

  // alocação e cópia dos dados
  cudaMalloc((void **) &d_a, size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  cudaMalloc((void **) &d_s, s_size);

  // definição do número de blocos e threads
  dim3 dimGrid(num_blocks,1,1);
  dim3 dimBlock(block_size,1,1);

  // chamada do kernel
  sum_cuda<<<dimGrid,dimBlock>>>(d_a, d_s, width);

  // cópia dos resultados para o host
  cudaMemcpy(s, d_s, s_size, cudaMemcpyDeviceToHost);

  // soma das reduções parciais
  for(int i = 1; i < num_blocks; i++) 
    s[0] += s[i];

  printf("\nSum = %f\n",s[0]);
  
  cudaFree(d_a);
  cudaFree(d_s);
}
