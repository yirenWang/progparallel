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

int main(int argc, char *argv[])
{
    unsigned long int size = atoi(argv[1]);
    unsigned long int iter = atoi(argv[2]);

    /* initialize random seed: */

    srand(time(NULL));

    // Création des données de travail (moyenne 3 vecteurs)
    float *A, *B, *C, *M, *M_simd;
    A = (float *)malloc(size * sizeof(float));
    B = (float *)malloc(size * sizeof(float));
    C = (float *)malloc(size * sizeof(float));
    M = (float *)malloc(size * sizeof(float));
    M_simd = (float *)malloc(size * sizeof(float));

    // remplir les vecteur avec des float aleatoire
    for (unsigned long i = 0; i < size; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
        C[i] = (float)(rand() % 360 - 180.0);
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

        for (unsigned long j = 0; j < size; j++)
        {
            M[j] = (A[j] + B[j] + C[j]) / 3;
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

        // each simd vector is size 8, we need to split the original vecteur into the appropriate size
        for (int i = 0; i < size / 8; i++)
        {
            __m256 a = _mm256_setr_ps(A[i * 8], A[i * 8 + 1], A[i * 8 + 2], A[i * 8 + 3], A[i * 8 + 4], A[i * 8 + 5], A[i * 8 + 6], A[i * 8 + 7]);
            __m256 b = _mm256_setr_ps(B[i * 8], B[i * 8 + 1], B[i * 8 + 2], B[i * 8 + 3], B[i * 8 + 4], B[i * 8 + 5], B[i * 8 + 6], B[i * 8 + 7]);
            __m256 c = _mm256_setr_ps(C[i * 8], C[i * 8 + 1], C[i * 8 + 2], C[i * 8 + 3], C[i * 8 + 4], C[i * 8 + 5], C[i * 8 + 6], C[i * 8 + 7]);
            // add a and b to m
            __m256 m = _mm256_add_ps(a, b);
            // add c and m to m
            m = _mm256_add_ps(m, c);
            // set a variable to 3 (we would like to divide by 3 for the average)
            a = _mm256_set1_ps(3);
            // division
            m = _mm256_div_ps(m, a);

            // mettre le resultat dans le vecteur global
            _mm256_storeu_ps((float *)(M_simd + (i * 8)), m);
        }
        t1_simd = std::chrono::high_resolution_clock::now();
        double duration_simd = std::chrono::duration<double>(t1 - t0).count();
        if (duration_simd < min_duration_simd)
            min_duration_simd = duration_simd;
    }

    // std::cout << "Total Time " << "cpp :" << std::endl;
    float ops = size;
    std::cout << size << " " << (min_duration / ops) << std::endl;
    std::cout << size << " " << (min_duration_simd / ops) << std::endl;

    for (int i = 0; i < size; i++)
    {
        if (M[i] != M_simd[i])
        {
            std::cout << "oops" << std::endl;
        }
    }

    free(A);
    free(B);
    free(C);
    free(M);

    return 0;
}