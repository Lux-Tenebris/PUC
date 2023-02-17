#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include<sys/time.h>

/*
 * sieve contém um erro no no segundo loop. i tem que ser igual à p*p, não p*2.
 * o array booleano 'prime' não foi liberado, podendo causar um leak de memória
 */
int sieve_of_eratosthenes(int n){

  int primes = 0; 
  int mem_size = n+1;
  bool *prime = (bool*) malloc(mem_size*sizeof(bool));
  int sqrt_n = sqrt(n);

  memset(prime, true, mem_size*sizeof(bool));

  for(int p = 2; p <= sqrt_n; p++){
    if(prime[p]) for(int i = p*p; i <= n; i += p) prime[i] = false;
  }

  for(int p = 2; p<=n; p++) primes += prime[p];

  free(prime);
  return(primes);
}

/*
 * O único número primo par é 2, logo vamos iterar apenas pelos números impares maiores que um 
 * Essa simples alteração nos economizou 50% da memória do algoritmo anterior, além de deixa o crivo duas vezes mais rápido
 */
int sieve_of_eratosthenes_ODD(int n){

  int primes = 0; 
  int mem_size = (n-1)/2;
  bool *prime = (bool*) malloc(mem_size*sizeof(bool));
  int sqrt_n = sqrt(n);

  memset(prime, true, mem_size*sizeof(bool));
  for(int p = 3; p <= sqrt_n; p += 2){ if(prime[p/2]) for(int i = p*p; i <= n; i += 2*p) prime[i/2] = false;
  }

  primes = n >= 2 ? 1 : 0;
  for(int p = 1; p <= mem_size; p++) primes += prime[p];

  free(prime);
  return(primes);
}

int sieve_of_eratosthenes_ODD_CP(int n){

  omp_set_num_threads(omp_get_num_procs()/2);

  int primes = 0; 
  int mem_size = (n-1)/2;
  bool *prime = (bool*) malloc(mem_size*sizeof(bool));
  int sqrt_n = sqrt(n);

  memset(prime, true, mem_size*sizeof(bool));

#pragma omp parallel for schedule(dynamic)
  for(int p = 3; p <= sqrt_n; p += 2){
    if(prime[p/2]) for(int i = p*p; i <= n; i += 2*p) prime[i/2] = false;
  }

  primes = n >= 2 ? 1 : 0;
#pragma omp parallel for reduction(+:primes)
  for(int p = 1; p <= mem_size; p++) primes += prime[p];

  free(prime);
  return(primes);
}

int main(){
  int n = 100000000;

  //printf("%d\n",sieve_of_eratosthenes(n));
  //printf("%d\n",sieve_of_eratosthenes_ODD(n));
  //
  struct timeval stop, start;
  gettimeofday(&start, NULL);
  printf("%d\n",sieve_of_eratosthenes(n)); // altere a funções para testar o tempo e o output das outras.
  gettimeofday(&stop, NULL);
  /* printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); */
  //printf("%lu\n",((stop.tv_usec) - (start.tv_usec)) / 1000);


  return 0;
}

//Tempos de Execução do Crivo de Erastotenes e suas variações
//
/*
   Esses testes foram executados na seguinte arquitetura

Architecture:            x86_64
CPU op-mode(s):        32-bit, 64-bit
Address sizes:         39 bits physical, 48 bits virtual
Byte Order:            Little Endian
CPU(s):                  8
On-line CPU(s) list:   0-7
Vendor ID:               GenuineIntel
Model name:            Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
CPU family:          6
Model:               142
Thread(s) per core:  2
Core(s) per socket:  4
Socket(s):           1
Stepping:            12
CPU(s) scaling MHz:  22%
CPU max MHz:         3900.0000
CPU min MHz:         400.0000
BogoMIPS:            3601.00
Virtualization features: 
Virtualization:        VT-x
Caches (sum of all):     
L1d:                   128 KiB (4 instances)
L1i:                   128 KiB (4 instances)
L2:                    1 MiB (4 instances)
L3:                    6 MiB (1 instance)
NUMA:                    
NUMA node(s):          1
NUMA node0 CPU(s):     0-7

Crivo de Erastotenes Padrão

real    0m1.402s
user    0m1.387s
sys     0m0.014s

Crivo de Erastotenes Impar

real    0m0.680s
user    0m0.653s
sys     0m0.026s

Crivo de Erastotenes Impar Paralelo

real    0m0.429s
user    0m1.651s
sys     0m0.010s
*/
