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

    // Création des données de travail (produit scalaire de A , B)
    float *A, *B;
    A = (float *)malloc(size * sizeof(float));
    B = (float *)malloc(size * sizeof(float));
    float S;
    float S_simd;

    // remplir les vecteur avec des float aleatoire
    for (unsigned long i = 0; i < size; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
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
        S = 0;
        t0 = std::chrono::high_resolution_clock::now();

        for (unsigned long j = 0; j < size; j++)
        {
            S += A[j] * B[j];
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1_simd - t0_simd).count();
        if (duration < min_duration)
            min_duration = duration;
    }

    // SIMD calculation
    for (auto it = 0; it < iter; it++)
    {
        S_simd = 0;
        t0_simd = std::chrono::high_resolution_clock::now();

        __m256 s = _mm256_set1_ps(0);
        // each simd vector is size 8, we need to split the original vecteur into the appropriate size
        for (int i = 0; i < size / 8; i++)
        {
            __m256 a = _mm256_loadu_ps(&A[i * 8]);
            __m256 b = _mm256_loadu_ps(&B[i * 8]);
            // multiple a and b (third term is a mask to specify how the answer is given)
            __m256 res = _mm256_dp_ps(a, b, 255);
            // add the resulting simd vector to the sum simd vector
            s = _mm256_add_ps(res, s);
        }
        // retrieve the sum into a vector

        float *res_simd;
        res_simd = (float *)malloc(size * sizeof(float));
        // mettre le resultat dans le vecteur global
        _mm256_storeu_ps((float *)(res_simd), s);
        S_simd = res_simd[0] + res_simd[4];

        t1_simd = std::chrono::high_resolution_clock::now();
        double duration_simd = std::chrono::duration<double>(t1 - t0).count();
        if (duration_simd < min_duration_simd)
            min_duration_simd = duration_simd;
    }

    // std::cout << "Total Time " << "cpp :" << std::endl;
    float ops = size;
    std::cout << "size : " << size << std::endl;
    std::cout << "temps scalaire " << (min_duration / ops) << std::endl;
    std::cout << "temps vectoriel " << (min_duration_simd / ops) << std::endl;

    std::cout << S << std::endl;
    std::cout << S_simd << std::endl;
    // if (S != S_simd)
    // {
    //     std::cout << S << std::endl;
    //     std::cout << S_simd << std::endl;
    // }
    free(A);
    free(B);

    return 0;
}
