#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

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

  for(int p = 3; p <= sqrt_n; p += 2){
    if(prime[p/2]) for(int i = p*p; i <= n; i += 2*p) prime[i/2] = false;
  }

  primes = n >= 2 ? 1 : 0;
  for(int p = 1; p <= mem_size; p++) primes += prime[p];

  free(prime);
  return(primes);
}

int sieve_of_eratosthenes_ODD_CP(int n){


  //omp_set_num_threads(2);

  int primes = 0; 
  int mem_size = (n-1)/2;
  bool *prime = (bool*) malloc(mem_size*sizeof(bool));
  int sqrt_n = sqrt(n);

  int pivot = sqrt_n / 2;
  if(pivot%2==0) pivot++;

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

  //
  //printf("%d\n",sieve_of_eratosthenes_ODD(n));
  //printf("%d\n",sieve_of_eratosthenes_ODD_CP(n));
  struct timeval stop, start;
  gettimeofday(&start, NULL);
  printf("%d\n",sieve_of_eratosthenes(n));
  gettimeofday(&stop, NULL);
  printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
  //printf("%lu\n",((stop.tv_usec) - (start.tv_usec)) / 1000);


  return 0;
} 
// Tempos de Execução do Crivo de Erastotenes e suas variações
/* 
Crivo de Erastotenes Padrão
  real    0m1.438s
  user    0m1.399s
  sys     0m0.037s
 Crivo de Erastotenes Impar
  real    0m0.768s
  user    0m0.753s
  sys     0m0.014s
 Crivo de Erastotenes Impar Paralelo
  real    0m0.305s
  user    0m2.306s
  sys     0m0.010s
*/
