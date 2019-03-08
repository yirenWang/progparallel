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
        if (A[i] % 2 == 0)
        {
            S[ne] = A[i];
            ne += 1;
        }
    }
    return ne;
}

unsigned long int parallele(int *A, int *S, unsigned long int size)
{
    unsigned long int ne = 0;

#pragma omp parallel for shared(ne)
    for (unsigned long int i = 0; i < size; i++)
    {
        if (A[i] % 2 == 0)
        {
#pragma omp atomic
            ne++;
            S[ne] = A[i];
        }
    }

    return ne;
}


unsigned long int parallele2(int *A, int *S, unsigned long int size)
{
    unsigned long int global_ne[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int* tmp = (int*)malloc(size*sizeof(int));

    #pragma omp parallel
    {
        int current_thread = omp_get_thread_num();
    #pragma omp for schedule(static, size/8)
        for(unsigned long int i=0; i < size; i++)
        {
            if (A[i] % 2 == 0)
            {   
                tmp[current_thread*size/8 + global_ne[current_thread]] = A[i];
                global_ne[current_thread] += 1;
            }
        }
    }
    

    int ne = 0;
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < global_ne[i]; j++)
        {   
            S[ne] = tmp[i*size/8 + j];
            ne++;
        }
    }
    free(tmp);
    return ne;
}

int main()
{

    /* initialize random seed: */
    srand(time(NULL));
    std::cout << "Size seq_duration / par_duration" << std::endl;

    for (unsigned long int size = 1024; size < (1024 * 1024 * 4); size *= 1.2)
    {
        unsigned long int iter = 256 * 1024 * 1024 / size;

        // Création des données de travail
        int *A, *S1, *S2;
        A = (int *)malloc(size * sizeof(int));
        S1 = (int *)malloc(size * sizeof(int));
        S2 = (int *)malloc(size * sizeof(int));

        unsigned long int ne1, ne2;
        for (unsigned long int i = 0; i < size; i++)
        {
            A[i] = (int)(rand() % 360 - 180.0);
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
            ne2 = parallele2(A, S2, size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double par_duration = std::chrono::duration<double>(t1 - t0).count();
        par_duration /= (size * iter);

        std::cout << size << " " << seq_duration / par_duration << std::endl;
        // std::cout << ne1 << " " << ne2  << std::endl;
        // std::cout << size << " " << seq_duration << " " << par_duration << std::endl;

        /*** Validation ***/
        // bool valide = false;
        // for (unsigned long int i = 0; i < size; i++)
        // {
        //     std::cout << S1[i] << " " << S2[i] << std::endl;
        //     if (S1[i] == S2[i])
        //     {
        //         valide = true;
        //     }
        //     else
        //     {
        //         valide = false;
        //         break;
        //     }
        // }

        // Libération de la mémoire : indispensable

        free(A);
        free(S1);
        free(S2);
    }
    return 0;
}

// Results

/*
(previous code  (cf state in gitlab))
Sequential
Time taken to fill : 0.0550597
Time taken to calculate : 0.0013971
Parallel
Time taken to fill : 0.223159
Time taken to calculate : 0.000986

Size: 1024 Iterations: 262144
Sequential
Time taken to fill : 0.000129
Time taken to calculate : 9.2e-06
Parallel
Time taken to fill : 0.0013003
Time taken to calculate : 1.22e-05
*/

// It takes longer to fill the vectors in parallel. It probably indicates a shared resource.
// There is a gain of 41% in parallel