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

    // Création des données de travail (filtrage gaussien de A)
    float *A, *S;
    A = (float *)malloc(size * sizeof(float));
    S = (float *)malloc(size * sizeof(float));

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
        S[0] = (2 * A[0] + A[1]) / 4;
        for (unsigned long j = 1; j < size-1; j++) {
            S[j] = (A[j-1] + 2*A[j] + A[j+1]) / 4;
        }
        S[size-1] = (2 * A[size-1] + A[size-2]) / 4;

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }

    // SIMD calculation
    for (auto it = 0; it < iter; it++)
    {
        t0_simd = std::chrono::high_resolution_clock::now();

        __m256 two = _mm256_set1_ps(2.0);
        __m256 four = _mm256_set1_ps(4.0);

        __m256 a0 = _mm256_loadu_ps(A);
        __m256 b0 = _mm256_loadu_ps(A+1);
        __m256 c0 = _mm256_set_ps(A[6], A[5], A[4], A[3], A[2], A[1], A[0], 0);

        // ( c0 + 2a0 + b0 ) / 4
        __m256 gauss0 = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(b0, _mm256_mul_ps(two, a0)), c0), four);
        _mm256_storeu_ps(S, gauss0);

        for (unsigned long j = 8; j < size; j+=8) {            
            __m256 a = _mm256_loadu_ps(A+j);
            __m256 b = _mm256_loadu_ps(A+j+1);
            __m256 c = _mm256_loadu_ps(A+j-1);
            
            // ( c + 2a + b ) / 4
            __m256 gauss = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(b, _mm256_mul_ps(two, a)), c), four);
            _mm256_storeu_ps(S+j, gauss);
        }

        __m256 af = _mm256_loadu_ps(A+size-8);
        __m256 bf = _mm256_set_ps(0, A[size-1], A[size-2], A[size-3], A[size-4], A[size-5], A[size-6], A[size-7]);
        __m256 cf = _mm256_loadu_ps(A+size-9);

        __m256 gaussf = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(bf, _mm256_mul_ps(two, af)), cf), four);
        _mm256_storeu_ps(S+size-8, gaussf);

        t1_simd = std::chrono::high_resolution_clock::now();

        double duration_simd = std::chrono::duration<double>(t1_simd - t0_simd).count();
        if (duration_simd < min_duration_simd)
            min_duration_simd = duration_simd;
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
