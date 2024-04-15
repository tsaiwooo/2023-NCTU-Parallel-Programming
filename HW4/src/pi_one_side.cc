#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int check(long long int *schedule, int size)
{
    int done = 0;
    for(int i=1 ; i<size ; i++){
        if(schedule[i] > 0){
            done++;
        }
    }
    return (done==(size-1));
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    long long int local_tosses = tosses / world_size;
    long long int local_hits = 0;
    long long int hits = 0;

    unsigned seed = (unsigned)time(NULL) + world_rank;
    srand(seed);

    for(long long int i=0; i<local_tosses ; ++i){
        double x = (double)rand_r(&seed)/RAND_MAX ;
        double y = (double)rand_r(&seed)/RAND_MAX ;
        double distance_squared = x*x + y*y;
        if(distance_squared<=1){
           local_hits++;
        }
    }

    // TODO: MPI init
    long long int *schedule;
    // MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &schedule);

    if (world_rank == 0)
    {
        // Master
        // long long int *oldschedule = malloc(size * sizeof(long long int));
        // Use MPI to allocate memory for the target window
        // long long int *schedule;

        // MPI_Alloc_mem(size * sizeof(int), MPI_INFO_NULL, &schedule);
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &schedule);

        for(int i=0 ; i<world_size ; ++i){
            schedule[i] = 0;
        }

        MPI_Win_create(schedule, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL,
          MPI_COMM_WORLD, &win);

        int ready = 0;
        while(!ready){
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = check(schedule, world_size);
            MPI_Win_unlock(0, win);
        }
        // MPI_Win_free(&win);

    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

       // Register with the master
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&local_hits, 1, MPI_LONG_LONG_INT, 0, world_rank, 1, MPI_LONG_LONG_INT, win);
        MPI_Win_unlock(0, win);

        MPI_Win_free(&win);
    }

    // MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        for(int i=1 ; i<world_size ; i++){
            local_hits += schedule[i];
        }
        pi_result = 4*local_hits / (double) tosses;
        MPI_Win_free(&win);
        MPI_Free_mem(schedule);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}