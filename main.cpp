#include <stdio.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define RAIO 8
float* grid;

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

    printf("Rank %d OK\n", rank);

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
    gpu_id = rank % num_gpus_available;

    cudaGetDeviceCount(&num_gpus_available);
    cudaSetDevice(gpu_id);
    printf("#%d - Malloc memory on GPU: %d (Size: %d)",rank, gpu_id, grid_size);
    cudaMalloc(&grid, grid_byte_size);
    



    printf("#%d - Freeing memory on GPU: %d", rank, gpu_id);
    cudaFree(grid);
    MPI_Finalize();
    return 0;
}