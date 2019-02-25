#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

void print_vector(float *v, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << v[i] << std::endl;
    }
}

float sequential(int size, int iter, float *A, float *B)
{
    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    double min_duration = DBL_MAX;

    float S;
    for (auto it = 0; it < iter; it++)
    {
        S = 0;
        t0 = std::chrono::high_resolution_clock::now();

        for (unsigned long j = 0; j < size; j++)
        {
            if (j < size / 10)

            {
                S += (A[j] * B[j]);
            }
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Time taken to calculate : " << min_duration << std::endl;
    return S;
}

float parallel(int size, int iter, float *A, float *B)
{
    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    float S;
    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();
        S = 0;

#pragma omp parallel for reduction(+ \
                                   : S)
        for (unsigned long j = 0; j < size; j++)
        {
            S += A[j] * B[j];
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Time taken to calculate : " << min_duration << std::endl;
    return S;
}

float parallel_schedule(int size, int iter, float *A, float *B)
{
    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    float S;
    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();
        S = 0;

#pragma omp parallel for schedule(dynamic, size / 100) reduction(+ \
                                                                 : S)
        for (unsigned long j = 0; j < size; j++)
        {
            if (j < size / 10)
            {
                S += A[j] * B[j];
            }
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Time taken to calculate : " << min_duration << std::endl;
    return S;
}

float parallel_no_reduction(int size, int iter, float *A, float *B)
{
    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;

    float total;
    int THREADS = 0;
#pragma omp parallel
    {
        THREADS = omp_get_num_threads();
    }

    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();

        float S[THREADS] = {0};

#pragma omp parallel for
        for (int k = 0; k < THREADS; k++)
        {
            for (unsigned long j = 0; j < size / THREADS; j++)
            {
                S[k] += A[j + k * size / THREADS] * B[j + k * size / THREADS];
            }
        }
        total = 0;
        for (int k = 0; k < THREADS; k++)
        {
            total += S[k];
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Time taken to calculate : " << min_duration << std::endl;
    return total;
}

int main(int argc, char *argv[])
{
    unsigned long int size = atoi(argv[1]);
    unsigned long int iter = atoi(argv[2]);

    /* initialize random seed: */

    srand(time(NULL));

    // Création des données de travail (moyenne 2 vecteurs)
    float *A, *B, S;
    A = (float *)malloc(size * sizeof(float));
    B = (float *)malloc(size * sizeof(float));
    S = 0;

    // Fill with random
    for (unsigned long i = 0; i < size; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
    }

    std::cout << "Sequential \n";
    float seq_S = sequential(size, iter, A, B);
    std::cout << seq_S << std::endl;
    std::cout << "Parallel \n";
    float para_S = parallel(size, iter, A, B);
    std::cout << para_S << std::endl;
    std::cout << "Parallel no reduction\n";
    float para_no_red_S = parallel_no_reduction(size, iter, A, B);
    std::cout << para_no_red_S << std::endl;
    std::cout << "Parallel schedule\n";
    float para_schedule_S = parallel_schedule(size, iter, A, B);
    std::cout << para_schedule_S << std::endl;
    // Parallel schedule DYNAMIC
    // Time taken to calculate : 0.0222222 (10 fois plus lent que scalaire)
    // Parallel schedule AUTO
    // Time taken to calculate : 0.0011869 (2 fois plus rapide que scalaire)
    // Parallel schedule STATIC, SIZE
    // Time taken to calculate : 0.0023483 (même temps que scalaire). C'est normal, il y a une thread qui fait tout le travail
    // Parallel schedule AUTO avec 10% de données traité
    // Time taken to calculate : 0.000778  (1.5 fois plus rapide que parallel reduction que 100% de données)
    // Time take sequential : 0.000227, Time taken parallel with reduction and schedule auto : 0.000683 (Seq 3 fois plus rapide que para)
    free(A);
    free(B);
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