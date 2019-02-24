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

    // Création des données de travail (minimum et maximum de A)
    float *A;
    A = (float *)malloc(size * sizeof(float));
    float min;
    float min_simd;
    float max;
    float max_simd;

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
        min = FLT_MAX;
        max = FLT_MIN;
        t0 = std::chrono::high_resolution_clock::now();

        for (unsigned long j = 0; j < size; j++)
        {
            if (min > A[j])
            {
                min = A[j];
            }
            if (max < A[j])
            {
                max = A[j];
            }
        }

        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }

    // SIMD calculation
    for (auto it = 0; it < iter; it++)
    {
        min_simd = FLT_MAX;
        max_simd = FLT_MIN;
        __m256 local_min = _mm256_set1_ps(FLT_MAX);
        __m256 local_max = _mm256_set1_ps(FLT_MIN);

        t0_simd = std::chrono::high_resolution_clock::now();

        // each simd vector is size 8, we need to split the original vecteur into the appropriate size
        for (int i = 0; i < size / 8; i++)
        {
            __m256 a = _mm256_loadu_ps(&A[i * 8]);
            // minimum term by term of the two vectors.
            local_min = _mm256_min_ps(a, local_min);
            local_max = _mm256_max_ps(a, local_max);
        }

        // store the mm256 into an array and iterate over it to find the max and the min

        float *min_vec = (float *)malloc(8 * sizeof(float));
        float *max_vec = (float *)malloc(8 * sizeof(float));
        _mm256_storeu_ps((float *)(min_vec), local_min);
        _mm256_storeu_ps((float *)(max_vec), local_max);

        // iteration
        for (int i = 0; i < 8; i++)
        {
            if (min_vec[i] < min_simd)
            {
                min_simd = min_vec[i];
            }

            if (max_vec[i] > max_simd)
            {
                max_simd = max_vec[i];
            }
        }
        t1_simd = std::chrono::high_resolution_clock::now();
        double duration_simd = std::chrono::duration<double>(t1_simd - t0_simd).count();
        if (duration_simd < min_duration_simd)
            min_duration_simd = duration_simd;
    }

    // std::cout << "Total Time " << "cpp :" << std::endl;
    float ops = size;
    std::cout << "Temps scalaire : " << (min_duration / ops) << std::endl;
    std::cout << "Temps vectorielle : " << (min_duration_simd / ops) << std::endl;

    std::cout << "max simd " << max_simd << std::endl;
    std::cout << "min simd " << min_simd << std::endl;
    std::cout << "max " << max << std::endl;
    std::cout << "min " << min << std::endl;

    free(A);
    return 0;
}