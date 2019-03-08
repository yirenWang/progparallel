#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

unsigned long int sequentiel(int *A, int *S, unsigned long int size)
{
    unsigned long int ne = 0;
    for (unsigned long int i = 0; i < size; i++)
    {
        if (A[i] % 2)
        {
            S[ne] = A[i];
            ne++;
        }
    }

    return ne;
}

unsigned long int sum(unsigned long int *S, unsigned long int size)
{
    unsigned long int s = 0;
    for (unsigned long int i = 0; i < size; i++)
    {
        s += S[i];
    }
    return s;
}

unsigned long int parallele(int *A, int *S, unsigned long int size)
{
    unsigned long int ne[4] = {0, 0, 0, 0};
    int *S_local = (int *)malloc(size * sizeof(int));
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
#pragma omp for schedule(static, size / 4)
        for (unsigned long int i = 0; i < size; i++)
        {
            if (A[i] % 2)
            {
                S_local[size / 4 * thread_num + ne[thread_num]] = A[i];
                ne[thread_num]++;
            }
        }
    }

    for (unsigned long int i = 0; i < 4; i++)
    {
        int offset = sum(ne, i);
        for (unsigned long int j = 0; j < ne[i]; j++)
        {
            S[j + offset] = S_local[i * size / 4 + j];
        }
    }

    free(S_local);
    return sum(ne, 4);
}

int main()
{

    /* initialize random seed: */
    srand(time(NULL));

    std::cout << "Vec size; Ratio; Seq duration; Par duration; Validité;" << std::endl;

    for (unsigned long int size = 1024; size < (1024 * 1024 * 4); size *= 1.2)
    {
        unsigned long int iter = 256 * 1024 * 1024 / size;

        // Création des données de travail
        int *A, *S1, *S2;
        int ne1, ne2;
        A = (int *)malloc(size * sizeof(int));
        S1 = (int *)malloc(size * sizeof(int));
        S2 = (int *)malloc(size * sizeof(int));

        for (unsigned long int i = 0; i < size; i++)
        {
            A[i] = (int)(rand() % 360 - 180);
        }

        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        double min_duration = DBL_MAX;
        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = 0; it < iter; it++)
        {
            ne1 = sequentiel(A, S1, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double seq_duration = std::chrono::duration<double>(t1 - t0).count();
        seq_duration /= (size * iter);

        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = 0; it < iter; it++)
        {
            ne2 = parallele(A, S2, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double par_duration = std::chrono::duration<double>(t1 - t0).count();
        par_duration /= (size * iter);

        /*** Validation ***/
        bool valide = false;
        if (ne1 == ne2)
        {
            valide = true;
            // for (unsigned long int j = 0; j < size; j++)
            // {
            //     std::cout << S1[j] << " " << S2[j] << std::endl;
            // }
        }
        else
        {
            valide = false;
        }

        std::cout << size << ";" << seq_duration / par_duration << ";" << seq_duration << ";" << par_duration << ";" << std::boolalpha << valide << std::endl;

        // Libération de la mémoire : indispensable

        free(A);
        free(S1);
        free(S2);
    }
    return 0;
}