#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>

void print_vector(float *v, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << v[i] << std::endl;
    }
}

void sequential(int size, int iter)
{
    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    // Création des données de travail (moyenne 2 vecteurs)
    float *A, *B, *M;
    A = (float *)malloc(size * sizeof(float));
    B = (float *)malloc(size * sizeof(float));
    M = (float *)malloc(size * sizeof(float));

    // Fill with random
    for (unsigned long i = 0; i < size; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
    }

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken to fill : " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
    double min_duration = DBL_MAX;

    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();

        for (unsigned long j = 0; j < size; j++)
        {
            M[j] = (A[j] + B[j]) / 2;
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Time taken to calculate : " << min_duration << std::endl;

    free(A);
    free(B);
    free(M);
}

void parallel(int size, int iter)
{
    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    // Création des données de travail (moyenne 2 vecteurs)
    float *A, *B, *M;
    A = (float *)malloc(size * sizeof(float));
    B = (float *)malloc(size * sizeof(float));
    M = (float *)malloc(size * sizeof(float));

// Fill with random
#pragma omp parallel for
    for (unsigned long i = 0; i < size; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
    }

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken to fill : " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
    double min_duration = DBL_MAX;

    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (unsigned long j = 0; j < size; j++)
        {
            M[j] = (A[j] + B[j]) / 2;
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Time taken to calculate : " << min_duration << std::endl;

    free(A);
    free(B);
    free(M);
}

int main(int argc, char *argv[])
{
    unsigned long int size = atoi(argv[1]);
    unsigned long int iter = atoi(argv[2]);

    /* initialize random seed: */

    srand(time(NULL));

    std::cout << "Sequential \n";
    sequential(size, iter);
    std::cout << "Parallel \n";
    parallel(size, iter);

    return 0;
}

// Results

/*

Sequential
Time taken to fill : 0.0550597
Time taken to calculate : 0.0013971
Parallel
Time taken to fill : 0.223159
Time taken to calculate : 0.000986

*/

// It takes longer to fill the vectors in parallel. It probably indicates a shared resource. There is an