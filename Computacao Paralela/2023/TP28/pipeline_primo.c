#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_N 100000
#define PIPE_MSG 0
#define END_MSG 1

int size;
int n;
int rank;

void removeMultipleOf3()
{
    int number, i;

    for (i = 1; i <= n / 2; i++)
    {
        number = 2 * i + 1;
        if (number > n)
            break;
        if (number % 3 > 0)
            MPI_Send(&number, 1, MPI_INT, 1, PIPE_MSG, MPI_COMM_WORLD);
    }
    number = -1; // Send termination signal
    MPI_Send(&number, 1, MPI_INT, 1, END_MSG, MPI_COMM_WORLD);
}

void removeMultipleOf5()
{
    int number;
    MPI_Status status;

    while (1)
    {
        MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == END_MSG)
            break;
        if (number % 5 > 0)
        {
            int next_rank = (rank + 1) % size; // Determine next rank in a circular manner
            MPI_Send(&number, 1, MPI_INT, next_rank, PIPE_MSG, MPI_COMM_WORLD);
        }
    }
    number = -1; // Send termination signal
    int next_rank = (rank + 1) % size; // Determine next rank in a circular manner
    MPI_Send(&number, 1, MPI_INT, next_rank, END_MSG, MPI_COMM_WORLD);
}

void countOnlyPrimes()
{
    int number, primeCount = 0, i, isComposite;
    MPI_Status status;

    while (1)
    {
        MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == END_MSG)
            break;
        if (number == -1)
            continue; // Skip termination signal
        isComposite = 0;
        for (i = 7; i * i <= number; i += 2)
        {
            if (number % i == 0)
            {
                isComposite = 1;
                break;
            }
        }
        if (!isComposite)
            primeCount++;
    }

    printf("Number of primes = %d\n", primeCount);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    n = atoi(argv[1]);
    if (n > MAX_N)
    {
        printf("N is too large. Maximum allowed value is %d\n", MAX_N);
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    switch (rank)
    {
    case 0:
        removeMultipleOf3();
        break;
    case 1:
        removeMultipleOf5();
        break;
    case 2:
        countOnlyPrimes();
        break;
    };

    MPI_Finalize();

    return 0;
}
