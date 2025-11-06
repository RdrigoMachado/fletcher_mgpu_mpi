#include "backend.hpp"
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>  // <-- para sleep()

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int rank;

    int comm_size;
    int num_gpus_available;
    int nx, ny, nz;
    int halo_size;
    int num_steps;
    int delta_time;
    int num_gpus;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    num_steps  = atoi(argv[4]);
    delta_time = atoi(argv[5]);
    num_gpus   = atoi(argv[6]);

    backend_init(rank, comm_size, nx, ny, nz, num_gpus);
    backend_data_initialize();
    backend_run(num_steps, delta_time);
    printf("Rank %d - comm_size %d\n", rank, comm_size);
    sleep(5);
    backend_finalize();

    MPI_Finalize();
    return 0;
}
