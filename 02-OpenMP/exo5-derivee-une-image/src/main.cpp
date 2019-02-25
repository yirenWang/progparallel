#include <iostream>
#include <chrono>

#include "structural_elements.hpp"
#include "container.hpp"
#include "global_parameters.hpp"
#include "ios.hpp"
#include "omp.h"

#define DEBUG 1

void sobel(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{

    for (auto i = 1; i < height - 1; i++)
    {
        for (auto j = 1; j < width - 1; j++)
        {
            if ((i == 0) || (i == height - 1) || (j == 0) || (j == width - 1))
            {
                Resultat[i][j] = 0;
            }
            else
            {
                Resultat[i][j] = std::abs(Source[i - 1][j - 1] + Source[i - 1][j] + Source[i - 1][j + 1] - (Source[i + 1][j - 1] + Source[i + 1][j] + Source[i + 1][j + 1]));
                Resultat[i][j] += std::abs(Source[i - 1][j - 1] + Source[i][j - 1] + Source[i + 1][j - 1] - (Source[i - 1][j + 1] + Source[i][j + 1] + Source[i + 1][j + 1]));
            }
        }
    }
}

void sobel_parallel(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{
#pragma omp parallel for
    for (auto i = 1; i < height - 1; i++)
    {
#pragma omp parallel for
        for (auto j = 1; j < width - 1; j++)
        {
            if ((i == 0) || (i == height - 1) || (j == 0) || (j == width - 1))
            {
                Resultat[i][j] = 0;
            }
            else
            {
                Resultat[i][j] = std::abs(Source[i - 1][j - 1] + Source[i - 1][j] + Source[i - 1][j + 1] - (Source[i + 1][j - 1] + Source[i + 1][j] + Source[i + 1][j + 1]));
                Resultat[i][j] += std::abs(Source[i - 1][j - 1] + Source[i][j - 1] + Source[i + 1][j - 1] - (Source[i - 1][j + 1] + Source[i][j + 1] + Source[i + 1][j + 1]));
            }
        }
    }
}
int main(int argc, char *argv[])
{
    unsigned int height, width;
    unsigned int nthreads = omp_get_max_threads();

    std::string image_filename(argv[1]);
    int ITER = atoi(argv[2]);

    get_source_params(image_filename, &height, &width);
    std::cout << "size: width*height" << width << " " << height << std::endl;
    u_char **Source, **Resultat;

    image<u_char> imgSource(height, width, &Source);
    image<u_char> imgResultat(height, width, &Resultat);

    auto fail = init_source_image(Source, image_filename, height, width);
    if (fail)
    {
        std::cout << "Chargement impossible de l'image" << std::endl;
        return 0;
    }

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    for (auto it = 0; it < ITER; it++)
    {
        sobel(Source, Resultat, height, width, nthreads);
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(t1 - t0).count() / ITER;

    std::cout << "time seq: " << duration << std::endl;

    t0 = std::chrono::high_resolution_clock::now();
    for (auto it = 0; it < ITER; it++)
    {
        sobel_parallel(Source, Resultat, height, width, nthreads);
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto duration_p = std::chrono::duration<double>(t1 - t0).count() / ITER;

    std::cout << "time parallel: " << duration_p << std::endl;

#ifdef DEBUG
    image_filename = std::string("Sobel.pgm");
    save_gray_level_image(&imgResultat, image_filename, height, width);
#endif

    return 0;
}
