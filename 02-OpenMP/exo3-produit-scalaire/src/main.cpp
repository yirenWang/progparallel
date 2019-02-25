#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

double sequentiel(double *A, double *B, unsigned long int size)
{
    double S = 0;
    for (unsigned long j = 0; j < size; j++)
    {
        if (j < size)

        {
            S += (A[j] * B[j]);
        }
    }
    return S;
}

double parallele(double *A, double *B, unsigned long int size)
{
    double S = 0;

#pragma omp parallel for reduction(+ \
                                   : S)
    for (unsigned long j = 0; j < size; j++)
    {
        S += A[j] * B[j];
    }

    return S;
}

int main()
{

    /* initialize random seed: */
    srand(time(NULL));

    for (unsigned long int size = 1024; size < (1024 * 1024 * 4); size *= 1.2)
    {
        unsigned long int iter = 256 * 1024 * 1024 / size;

        // Création des données de travail
        double *A, *B, *C;
        A = (double *)malloc(size * sizeof(double));
        B = (double *)malloc(size * sizeof(double));

        double S1, S2;

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
            S1 = sequentiel(A, B, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double seq_duration = std::chrono::duration<double>(t1 - t0).count();
        seq_duration /= (size * iter);

        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = 0; it < iter; it++)
        {
            S2 = parallele(A, B, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double par_duration = std::chrono::duration<double>(t1 - t0).count();
        par_duration /= (size * iter);

        std::cout << size << " " << seq_duration / par_duration << std::endl;
        // std::cout << size << " " << seq_duration << " " << par_duration << std::endl;

        /*** Validation ***/
        bool valide = false;

        if (S1 == S2)
        {
            valide = true;
        }
        else
        {
            valide = false;
            break;
        }

        // Libération de la mémoire : indispensable

        free(A);
        free(B);
    }
    return 0;
}
