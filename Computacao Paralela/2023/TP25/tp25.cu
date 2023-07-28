/*
==22697== NVPROF is profiling process 22697, command: ./cuda
==22697== Profiling application: ./cuda

a[39999999] = 799999980000000.000000
==22697== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 51.58%  465.93ms         2  232.97ms  452.49us  465.48ms  [CUDA memcpy HtoD]
 41.60%  375.74ms         2  187.87ms  362.38us  375.38ms  [CUDA memcpy DtoH]
  5.15%  46.538ms         1  46.538ms  46.538ms  46.538ms  scan_cuda(double*, double*, int)
  1.67%  15.115ms         1  15.115ms  15.115ms  15.115ms  add_cuda(double*, double*, int)

==22697== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 79.36%  904.34ms         4  226.08ms  90.997us  465.75ms  cudaMemcpy
 20.50%  233.60ms         2  116.80ms  8.9960us  233.59ms  cudaMalloc
  0.06%  705.57us         2  352.78us  34.170us  671.40us  cudaFree
  0.04%  484.99us        90  5.3880us     270ns  207.45us  cuDeviceGetAttribute
  0.02%  261.25us         2  130.62us  28.939us  232.31us  cudaLaunch
  0.01%  96.956us         1  96.956us  96.956us  96.956us  cuDeviceTotalMem
  0.01%  65.796us         1  65.796us  65.796us  65.796us  cuDeviceGetName
  0.00%  13.286us         6  2.2140us     349ns  10.129us  cudaSetupArgument
  0.00%  6.1640us         2  3.0820us  1.4660us  4.6980us  cudaConfigureCall
  0.00%  2.7220us         2  1.3610us  1.0200us  1.7020us  cuDeviceGetCount
  0.00%  1.0530us         2     526ns     517ns     536ns  cuDeviceGet
---------------------------------------------------------------------------------
a[39999999] = 799999980000000.000000

real    0m0.507s
user    0m0.211s
sys     0m0.291s

O motivo da versão de gpu demorar mais, é porque a alocação de memória leva muito tempo. Porém, observando as linhas 10 e 11, vemos que a execução em si, é mais rápida que os resultados sequenciais na linha 29
*/

#include <stdio.h>
#include <stdlib.h>

__global__ void scan_cuda(double* a, double *s, int width) {
  int t = threadIdx.x;
  int b = blockIdx.x*blockDim.x;
  double x;
  __shared__ double p[1024];
  if(b+t < width)
    p[t] = a[b+t];
  __syncthreads();

  for (int i = 1; i < blockDim.x; i *= 2) {
    if(t >= i)
      x = p[t] + p[t-i];
    __syncthreads();
    if(t >= i)
      p[t] = x;
    __syncthreads();
  }
  if(b + t < width)
    a[b+t] = p[t];
  if(t == blockDim.x-1)
    s[blockIdx.x+1] = a[b+t];
}

__global__ void add_cuda(double *a, double *s, int width) {
  int t = threadIdx.x;
  int b = blockIdx.x*blockDim.x;
  if(b+t < width)
    a[b+t] += s[blockIdx.x];
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
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_s, s_size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  dim3 dimBlock(block_size, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);

  scan_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);

  cudaMemcpy(s, d_s, s_size, cudaMemcpyDeviceToHost);

  s[0] = 0;
  for (int i = 1; i < num_blocks; i++)
    s[i] += s[i-1];

  cudaMemcpy(d_s, s, s_size, cudaMemcpyHostToDevice);

  add_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);

  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

  printf("\na[%d] = %f\n",width-1,a[width-1]);

  cudaFree(d_a);
  cudaFree(d_s);
}
