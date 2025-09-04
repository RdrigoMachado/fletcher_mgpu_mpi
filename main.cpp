#include "backend.hpp"
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>  // <-- para sleep()

int get_gpu_id(){
    int local_rank;
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    return local_rank;
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int rank;

    int comm_size;
    int num_gpus_available;
    int nx, ny, nz;
    int sx, sy, sz;
    int halo_size;
    int grid_size;
    int grid_byte_size;
    int num_steps;
    int num_gpus;
    int gpu_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    num_steps = atoi(argv[4]);
    num_gpus = atoi(argv[5]);
    sx = RAIO + nx + RAIO;
    sy = RAIO + ny + RAIO;
    sz = RAIO + (nz / num_gpus) + RAIO;
    halo_size = sx * sy * RAIO;
    grid_size = sx * sy * sz;
    grid_byte_size = grid_size * sizeof(float);

    gpu_id = get_gpu_id();

    backend_init(gpu_id);
    {
        sleep(10);
    }
    backend_finalize();
    MPI_Finalize();
    return 0;
}
