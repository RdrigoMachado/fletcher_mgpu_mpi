#include "backend.hpp"
#include <cuda_runtime.h>

cudaStream_t streams[2];
float *buffer_out[2];
float *buffer_in[2];
int sxsysz;
int sx, sy, sz_slice;
int process_num;
int universe_size;

float* dev_ch1dxx;
float* dev_ch1dyy;
float* dev_ch1dzz;
float* dev_ch1dxy;
float* dev_ch1dyz;
float* dev_ch1dxz;
float* dev_v2px;
float* dev_v2pz;
float* dev_v2sz;
float* dev_v2pn;
float* dev_pp;
float* dev_pc;
float* dev_qp;
float* dev_qc;


void backend_init(int rank, int comm_size, int sx, int sy, int sz){
    int deviceCount, gpu_id;
    int first = 0, last = comm_size - 1;
    process_num     = rank;
    universe_size   = comm_size;

    cudaGetDeviceCount(&deviceCount);
    gpu_id = rank % deviceCount;
    cudaSetDevice(gpu_id);
    cudaStreamCreate(&streams[HALO]);
    cudaStreamCreate(&streams[COMPUTE]);

    if(rank == first){
        sxsysz = sx * sy * (ABSORB + BOARD_SIZE + sz);
    } else if (rank == last){
        sxsysz = sx * sy * (sz + BOARD_SIZE + ABSORB);
    } else {
        sxsysz = sx * sy * (BOARD_SIZE + sz + BOARD_SIZE);
    }
    cudaMallocHost(&buffer_out[LEFT],  sxsysz * sizeof(float));
    cudaMallocHost(&buffer_in[LEFT],   sxsysz * sizeof(float));

    cudaMallocHost(&buffer_out[RIGHT], sxsysz * sizeof(float));
    cudaMallocHost(&buffer_in[RIGHT],  sxsysz * sizeof(float));
}

void backend_data_initialize(int sx, int sy, int sz){
    cudaMalloc(&dev_ch1dxx, sxsysz * sizeof(float));
    cudaMalloc(&dev_ch1dyy, sxsysz * sizeof(float));
    cudaMalloc(&dev_ch1dzz, sxsysz * sizeof(float));
    cudaMalloc(&dev_ch1dxy, sxsysz * sizeof(float));
    cudaMalloc(&dev_ch1dyz, sxsysz * sizeof(float));
    cudaMalloc(&dev_ch1dxz, sxsysz * sizeof(float));
    cudaMalloc(&dev_v2px,   sxsysz * sizeof(float));
    cudaMalloc(&dev_v2pz,   sxsysz * sizeof(float));
    cudaMalloc(&dev_v2sz,   sxsysz * sizeof(float));
    cudaMalloc(&dev_v2pn,   sxsysz * sizeof(float));
    cudaMalloc(&dev_pp,     sxsysz * sizeof(float));
    cudaMalloc(&dev_pc,     sxsysz * sizeof(float));
    cudaMalloc(&dev_qp,     sxsysz * sizeof(float));
    cudaMalloc(&dev_qc,     sxsysz * sizeof(float));
}

void backend_run(int num_steps){
    // insert_source();
    // swap_borders(int universe_size, int process_num, );
}

void backend_finalize(){
    cudaStreamDestroy(streams[HALO]);
    cudaStreamDestroy(streams[COMPUTE]);
    cudaFree(dev_ch1dxx);
    cudaFree(dev_ch1dyy);
    cudaFree(dev_ch1dzz);
    cudaFree(dev_ch1dxy);
    cudaFree(dev_ch1dyz);
    cudaFree(dev_ch1dxz);
    cudaFree(dev_v2px);
    cudaFree(dev_v2pz);
    cudaFree(dev_v2sz);
    cudaFree(dev_v2pn);
    cudaFree(dev_pp);
    cudaFree(dev_pc);
    cudaFree(dev_qp);
    cudaFree(dev_qc);
}
