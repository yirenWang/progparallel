#include <chrono>
#include <random>
#include <iostream>
#include <float.h>

int sum_array(int *array, int size)
{
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum;
}

void print_array(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int time_taken = 0;
    int no_philo = 7;
    int *philo, *baguette;
    philo = (int *)malloc(no_philo * sizeof(int));
    baguette = (int *)malloc(no_philo * sizeof(int));

    while (time_taken < 120)
    {
        for (int i = 0; i < no_philo - 1; i++)
        {
            // if the philo didn't eat yet and check if the baguette are being used
            if (philo[i] == 0)
            {
                if (baguette[i] + baguette[i + 1] == 0)
                {
                    printf("philo %d start to eat \n", i);
                    // baguette is free so he eats
                    philo[i] = 1;
                    baguette[i] = 1;
                    baguette[i + 1] = 1;
                }
            }
            else if (philo[i] < 15) // eat
            {
                printf("philo %d is eating \n", i);
                philo[i] += 1;
            }
            else if (philo[i] == 15) // finished eating
            {
                printf("philo %d finished eating \n", i);
                // frees the baguette
                baguette[i] = 0;
                baguette[i + 1] = 0;
                philo[i] += 1;
            }
        }
        // special case for the last philo
        if (philo[no_philo - 1] == 0 && baguette[0] == 0 && baguette[no_philo - 1] == 0)
        {
            // you can eat
            philo[no_philo - 1] = 1;
            baguette[no_philo - 1] = 1;
            baguette[0] = 1;
        }
        else if (philo[no_philo - 1] < 15) // eat
        {
            printf("philo %d is eating \n", no_philo - 1);
            philo[no_philo - 1] += 1;
        }
        else if (philo[no_philo - 1] == 15) // finished eating
        {
            printf("philo %d finished eating \n", no_philo);
            // frees the baguette
            baguette[no_philo - 1] = 0;
            baguette[0] = 0;
            philo[no_philo - 1] += 1;
        }

        time_taken += 1;
        printf("State of philosophers: ");
        print_array(philo, no_philo);
        printf("State of baguettes: ");
        print_array(baguette, no_philo);
    }
}