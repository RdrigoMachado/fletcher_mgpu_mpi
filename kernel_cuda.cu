#include "kernel_cuda.hpp"

__global__ void kernel_insert_source(const float val, float *qp, float *qc, int offset) {
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[offset]+=val;
    qc[offset]+=val;
  }
}
void insert_source(const float val, float *qp, float *qc, int offset) {
    dim3 threadsPerBlock(BSIZE_X, 1);
    dim3 numBlocks(1,1);
    kernel_insert_source<<<numBlocks, threadsPerBlock>>>(val, qp, qc, offset);
}



__global__ void kernel_setHaloWithPartitionSource(int partition_id, int halo_size,
    int halo_recv_front_offset, int halo_recv_back_offset, int halo_send_front_offset, int halo_send_back_offset,
    float *pp){
    // for(int i = 0; i < halo_size; i++){
    //     pp[halo_recv_front_offset] = 1;
    //     pp[halo_recv_back_offset] = 1;
    //     pp[halo_send_front_offset] =  1;
    //     pp[halo_send_back_offset]  =  1;
    // }
    for(int i = 0; i < halo_size; i++){
        pp[halo_recv_front_offset + i] = 6;
        pp[halo_recv_back_offset + i] = 9;
        pp[halo_send_front_offset + i] =  1;
        pp[halo_send_back_offset + i]  =  1;
    }
}

void setHaloWithPartitionSource(int partition_id, int halo_size,
    int halo_recv_front_offset, int halo_recv_back_offset, int halo_send_front_offset, int halo_send_back_offset,
    float *pp){

        kernel_setHaloWithPartitionSource<<<1, 1>>>(partition_id, halo_size,
            halo_recv_front_offset, halo_recv_back_offset, halo_send_front_offset, halo_send_back_offset, pp);
        cudaDeviceSynchronize();
}
