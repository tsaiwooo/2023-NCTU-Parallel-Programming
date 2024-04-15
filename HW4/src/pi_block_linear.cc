#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);

    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---


    // TODO: init MPI   

    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    long long int local_tosses = tosses / world_size;
    long long int local_hits = 0;
    long long int hits = 0;

    unsigned seed = (unsigned)time(NULL) + world_rank;
    srand(seed);
    if (world_rank > 0)
    {
        // TODO: handle workers
        // unsigned seed = (unsigned)time(NULL) + world_rank;
        // srand(seed);

        for(long long int i=0; i<local_tosses ; ++i){
            double x = (double)rand_r(&seed)/RAND_MAX ;
            double y = (double)rand_r(&seed)/RAND_MAX ;
            double distance_squared = x*x + y*y;
            if(distance_squared<=1){
               local_hits++;
            }
        }
        MPI_Send(&local_hits,1,MPI_LONG_LONG,0,0,MPI_COMM_WORLD);

    }
    else if (world_rank == 0)
    {
        // printf("msater in\n");
        // TODO: master
        // unsigned seed = (unsigned)time(NULL) + world_rank;
        // srand(seed);

        for(long long int i=0; i<local_tosses ; ++i){
            double x = (double)rand_r(&seed)/RAND_MAX ;
            double y = (double)rand_r(&seed)/RAND_MAX ;
            double distance_squared = x*x + y*y;
            if(distance_squared<=1){
               hits++;
            }
        }
        for(int i=1 ; i<world_size ; ++i){
            long long int receive;
            MPI_Recv(&receive,1,MPI_LONG_LONG,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            hits += receive;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = 4*hits / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
