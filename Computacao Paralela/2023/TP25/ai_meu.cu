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
