#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10
#define MAX 4
#define NUMBER 3

int main(int argc, char* argv[]) {
  int p, rank, largest_partial, largest_end, numProcs;
  int buffer[N];
  int i;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  if (rank == 0) {
    srand(time(NULL));
    for (i = 0; i < N; i++) {
      buffer[i] = rand() % MAX;
    }
  }

  MPI_Bcast(buffer, N, MPI_INT, 0, MPI_COMM_WORLD);

  int start = rank * (N / numProcs);
  int end = (rank + 1) * (N / numProcs);
  largest_partial = buffer[start];
  for (i = start + 1; i < end; i++) {
    if (buffer[i] > largest_partial) {
      largest_partial = buffer[i];
    }
  }

  MPI_Reduce(&largest_partial, &largest_end, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Largest element: %d\n", largest_end);
  }

  MPI_Finalize();
}
