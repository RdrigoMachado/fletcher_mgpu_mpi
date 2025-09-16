#ifdef BACKEND_KOKKOS
#include <Kokkos_Core.hpp>
#endif

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
    int sx, sy, sz;
    int halo_size;
    int grid_size;
    int grid_byte_size;
    int num_steps;
    int num_gpus;
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

#ifdef BACKEND_KOKKOS
    auto settings = Kokkos::InitializationSettings();
    Kokkos::initialize(settings);
    {
#endif

    backend_init(rank, comm_size, sx, sy, sz);
    sleep(5);
    // backend_finalize();

#ifdef BACKEND_KOKKOS
    }
    Kokkos::finalize();
#endif

    MPI_Finalize();
    return 0;
}
