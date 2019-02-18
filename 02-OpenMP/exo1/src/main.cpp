#include <iostream>
#include <omp.h>

// Comment out each section for tests
int main()
{
#pragma omp parallel
    {
        // on récupère le numéro du thread
        int i = omp_get_thread_num();

        // No further instructions
        std::cout << "Th read #" << i << "says hello ! \n"
                  << std::endl;

        // Example of output
        /* 
        Th read #Th read #0says hello !

        1says hello !
        */
        // TIt's normal, there's no reason for them not to be executed at the same time. My computer has 2 threads

#pragma omp critical
        std::cout << "Thread #" << i << " says hello ! \n";
        // Example of output
        /*
        Thread #0 says hello !
        Thread #1 says hello !
        */
        // Yay : critical ensures that the code is executed sequentially.
    }
    return 0;
}