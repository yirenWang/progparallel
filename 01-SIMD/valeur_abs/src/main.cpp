#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <math.h>

void print_vector(float *v, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << v[i] << std::endl;
    }
}

int main(int argc, char *argv[])
{
    unsigned long int size = atoi(argv[1]);
    unsigned long int iter = atoi(argv[2]);

    /* initialize random seed: */

    srand(time(NULL));

    // Création des données de travail (valeur abs de A)
    float *A, *S, *S_simd;
    A = (float *)malloc(size * sizeof(float));
    S = (float *)malloc(size * sizeof(float));
    S_simd = (float *)malloc(size * sizeof(float));

    // remplir les vecteur avec des float aleatoire
    for (unsigned long i = 0; i < size; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
    }

    // define the timers
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::chrono::high_resolution_clock::time_point t0_simd = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1_simd = std::chrono::high_resolution_clock::now();

    double min_duration = DBL_MAX;
    double min_duration_simd = DBL_MAX;

    // normal calculation
    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();
        for (unsigned long j = 0; j < size-1; j++) {
            S[j] = fabsf(A[j]);
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }

    // SIMD calculation
    for (auto it = 0; it < iter; it++)
    {
        t0_simd = std::chrono::high_resolution_clock::now();
        __m256 zero = _mm256_set1_ps(0.0);
        for (unsigned long j = 0; j < size; j+=8) {            
            __m256 a = _mm256_loadu_ps(A+j);
            __m256 b = _mm256_sub_ps(zero, a);

            _mm256_storeu_ps(S_simd+j, _mm256_max_ps(a, b));
        }

        t1_simd = std::chrono::high_resolution_clock::now();

        double duration_simd = std::chrono::duration<double>(t1_simd - t0_simd).count();
        if (duration_simd < min_duration_simd)
            min_duration_simd = duration_simd;
    }

    /* validation */
    bool valid = true;
    for (int i = 0; i < size; i++)
    {
        if (S[i] != S_simd[i])
        {
            valid = false;
        }
    }

    // std::cout << "Total Time " << "cpp :" << std::endl;
    float ops = size;
    std::cout << "size : " << size << std::endl;
    std::cout << "temps scalaire " << (min_duration / ops) << std::endl;
    std::cout << "temps vectoriel " << (min_duration_simd / ops) << std::endl;

    free(A);
    free(S);

    return 0;
}