#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

void sequentiel(double *A, double *B, double *S, unsigned long int size)
{
    for (unsigned long int i = 0; i < size; i++)
    {
        S[i] = (A[i] + B[i]) / 2;
    }
}

void parallele(double *A, double *B, double *S, unsigned long int size)
{
    double min_duration = DBL_MAX;

#pragma omp parallel for
    for (unsigned long j = 0; j < size; j++)
    {
        S[j] = (A[j] + B[j]) / 2;
    }
}

int main()
{

    /* initialize random seed: */
    srand(time(NULL));
    std::cout << "Size seq_duration / par_duration" << std::endl;

    for (unsigned long int size = 1024; size < (1024 * 1024 * 4); size *= 1.2)
    {
        unsigned long int iter = 256 * 1024 * 1024 / size;

        // Création des données de travail
        double *A, *B, *C, *S1, *S2;
        A = (double *)malloc(size * sizeof(double));
        B = (double *)malloc(size * sizeof(double));
        S1 = (double *)malloc(size * sizeof(double));
        S2 = (double *)malloc(size * sizeof(double));

        for (unsigned long int i = 0; i < size; i++)
        {
            A[i] = (double)(rand() % 360 - 180.0);
            B[i] = (double)(rand() % 360 - 180.0);
        }

        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        double min_duration = DBL_MAX;
        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = 0; it < iter; it++)
        {
            sequentiel(A, B, S1, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double seq_duration = std::chrono::duration<double>(t1 - t0).count();
        seq_duration /= (size * iter);

        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = 0; it < iter; it++)
        {
            parallele(A, B, S2, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double par_duration = std::chrono::duration<double>(t1 - t0).count();
        par_duration /= (size * iter);

        std::cout << size << " " << seq_duration / par_duration << std::endl;
        // std::cout << size << " " << seq_duration << " " << par_duration << std::endl;

        /*** Validation ***/
        bool valide = false;
        for (unsigned long int i = 0; i < size; i++)
        {
            if (S1[i] == S2[i])
            {
                valide = true;
            }
            else
            {
                valide = false;
                break;
            }
        }

        // Libération de la mémoire : indispensable

        free(A);
        free(B);
        free(S1);
        free(S2);
    }
    return 0;
}

// Results

/*
(previous code  (cf state in gitlab))
Sequential
Time taken to fill : 0.0550597
Time taken to calculate : 0.0013971
Parallel
Time taken to fill : 0.223159
Time taken to calculate : 0.000986

Size: 1024 Iterations: 262144
Sequential
Time taken to fill : 0.000129
Time taken to calculate : 9.2e-06
Parallel
Time taken to fill : 0.0013003
Time taken to calculate : 1.22e-05
*/

// It takes longer to fill the vectors in parallel. It probably indicates a shared resource.
// There is a gain of 41% in parallel