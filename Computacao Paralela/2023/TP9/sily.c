#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>

/* input: 10000 
 * Sequencial
real    0m0.241s 
user    0m0.240s 
sys     0m0.001s 
 * Parallel
real    0m0.063s
user    0m0.468s
sys     0m0.004s
input: 100000 
 * Sequencial
real    0m21.477s
user    0m21.400s
sys     0m0.004s
 * Parallel
real    0m5.500s
user    0m40.119s
sys     0m0.003s
*/

int main() 
{
   int i, j, n = 100000; 

   // Allocate input, output and position arrays
   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));   
   int *out = (int*) calloc(n, sizeof(int));   

   // Initialize input array in the reverse order
   for(i=0; i < n; i++)
      in[i] = n-i;  

    
   // Silly sort (you have to make this code parallel)
   #pragma omp parallel for private(j)
   for(i=0; i < n; i++) 
      for(j=0; j < n; j++)
         if(in[i] > in[j]) 
            pos[i]++;	

   // Move elements to final position
   memset(out, 0, n*sizeof(int));  // initialize out array to zeros
   for(i=0; i < n; i++) 
      out[pos[i]] = in[i];
   
   // print output array
    /* for(i=0; i < n; i++)  */
    /*    printf("%d ",out[i]); */

   // Check if answer is correct
   for(i=0; i < n; i++)
      if(i+1 != out[i]) 
      {
         printf("test failed\n");
         exit(0);
      }

   printf("test passed\n"); 

   // Free memory
   free(in);
   free(pos);
   free(out);
   return 0;
}  
