#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define THREADS_MAX 8

void* thread_function(void* args)
{
    int *toss_times = (int*) args;
    int *targets = malloc(sizeof(int) * 1);
    
    unsigned int seed = (unsigned int)pthread_self();
    srand(seed);

    register int temp = 0;
    register float x, y, distance_squared;

    for (int toss = 0; toss < *toss_times; toss++) {
        // printf("%d\n",*toss_times);
        x = (float) rand() / RAND_MAX;
        y = (float) rand() / RAND_MAX;

        distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            temp += 1;
    }

    targets[0] = temp;
    pthread_exit((void*)targets);
}

int main(int argc, char argv[]) 
{
    int number_of_tosses = 12500000;
    int number_in_circle = 0;

    pthread_t thread[THREADS_MAX];
    void* retvals[THREADS_MAX];

    // create pthread
    for(size_t i = 0; i < THREADS_MAX; i++) {
        pthread_create(&thread[i], NULL, thread_function, (void*)&number_of_tosses);
    }

    // pthread join
    for (int i = 0; i < THREADS_MAX; i++)
    {
        pthread_join(thread[i], &retvals[i]);
        number_in_circle += *(int*)retvals[i];
    }

    // final estimate value
    float pi_estimate = 4 * number_in_circle /((float) number_of_tosses * THREADS_MAX);
    printf("%f\n", pi_estimate);

    return 0;
}