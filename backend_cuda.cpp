#include "backend.hpp"
#include <cuda_runtime.h>

cudaStream_t streams[2];

void backend_init(int gpu_id){
    cudaSetDevice(gpu_id);
    cudaStreamCreate(&streams[HALO]);
    cudaStreamCreate(&streams[COMPUTE]);
}

void backend_data_initialize(){

}

void backend_run(int num_steps){

}

void backend_finalize(){
    cudaStreamDestroy(streams[HALO]);
    cudaStreamDestroy(streams[COMPUTE]);
}
