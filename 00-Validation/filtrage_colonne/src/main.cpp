#include <chrono>
#include <random>
#include <iostream>
#include <float.h>

int main(int argc, char *argv[])
{
    unsigned long int size = atoi(argv[1]);
    unsigned long int height = atoi(argv[2]);
    unsigned long int iter = atoi(argv[3]);

    /* initialize random seed: */

    srand(time(NULL));

    // Création des données de travail
    float *A, *S;
    A = (float *)malloc(size * height * sizeof(float));
    S = (float *)malloc(size * height * sizeof(float));

    for (unsigned long i = 0; i < size * height; i++)
    {
        A[i] = (float)(rand() % 360 - 180.0);
    }

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;

    for (auto it = 0; it < iter; it++)
    {
        t0 = std::chrono::high_resolution_clock::now();
        for (unsigned long j = 0; j < (size * height) - 1; j++)
        {
            S[j] = A[j] + A[j + 1];
        }
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        if (duration < min_duration)
            min_duration = duration;
    }

    // std::cout << "Total Time " << "cpp :" << std::endl;
    float ops = size * (height - 1);
    std::cout << size << " " << (min_duration / ops) << std::endl;
    free(A);
    free(S);

    return 0;
}
