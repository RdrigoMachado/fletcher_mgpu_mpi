#include "backend.hpp"
#include "kokkos_definitions.hpp"

ExecSpaceType exec_spaces[2];

void backend_init(int gpu_id){
    Kokkos::InitArguments args;
    args.device_id = gpu_id;
    Kokkos::initialize(args);
    #ifdef KOKKOS_ENABLE_CUDA
        streamType stream_halo, stream_compute;
        cudaSetDevice(gpu_id);
        cudaStreamCreate(&stream_halo);
        cudaStreamCreate(&stream_compute);
        exec_spaces[HALO] = ExecSpaceType(stream_halo);
        exec_spaces[COMPUTE] = ExecSpaceType(stream_compute);
    #elif defined(KOKKOS_ENABLE_HIP)
        streamType stream_halo, stream_compute;
        hipSetDevice(gpu_id);
        hipStreamCreate(&stream_halo);
        hipStreamCreate(&stream_compute);
        exec_spaces[HALO] = ExecSpaceType(stream_halo);
        exec_spaces[COMPUTE] = ExecSpaceType(stream_compute);
    #endif

}

void backend_data_initialize(){
}

void backend_run(int num_steps){

}

void backend_finalize(){
    #ifdef KOKKOS_ENABLE_CUDA
        cudaStreamDestroy(exec_spaces[HALO].cuda_stream());
        cudaStreamDestroy(exec_spaces[COMPUTE].cuda_stream());
    #elif defined(KOKKOS_ENABLE_HIP)
        hipStreamDestroy(exec_spaces[HALO].hip_stream());
        hipStreamDestroy(exec_spaces[COMPUTE].hip_stream());
    #endif
        Kokkos::finalize();
}
