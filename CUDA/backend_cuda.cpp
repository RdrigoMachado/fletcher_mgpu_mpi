#include "backend.hpp"
#include "kernel_cuda.hpp"
#include <bits/types/locale_t.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>

cudaStream_t streams[2];


float *buffer_out[2];
float *buffer_in[2];

int total_partitions;
int partition_id;
int sx, sy, sz, sz_local;
int local_grid_size;
int halo_size;
int absorb_size;
bool is_source_gpu = false;
int offset = 0;

int halo_recv_front_offset;
int halo_recv_back_offset;
int halo_send_front_offset;
int halo_send_back_offset;

float *halo_recv_front;
float *halo_recv_back;
float *halo_send_front;
float *halo_send_back;

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
float* host_grid;


void compute_grid_dimensions_and_offsets(int partition_id, int total_partitions,int nx, int ny, int nz){
    int last = total_partitions - 1;
    int local_nz = nz / total_partitions;
    is_source_gpu = (total_partitions / 2) == partition_id ? true: false;

    sx = ABSORB + HALO + nx + HALO + ABSORB;
    sy = ABSORB + HALO + ny + HALO + ABSORB;
    sz = ABSORB + HALO + nz + HALO + ABSORB;
    halo_size = sx * sy * HALO;
    absorb_size = sx * sy * ABSORB;

    if(SINGLE_GRID == total_partitions){
        sz_local = ABSORB + HALO + local_nz + HALO + ABSORB;
        local_grid_size = sx * sy * sz_local;

        halo_recv_front_offset = absorb_size;
        halo_send_front_offset = halo_recv_front_offset + halo_size;

        halo_recv_back_offset  = local_grid_size - halo_size - absorb_size;
        halo_send_back_offset  = halo_recv_back_offset - halo_size;

    } else if(FIRST_GRID == partition_id){
        sz_local = ABSORB + HALO + local_nz + HALO;
        local_grid_size = sx * sy * sz_local;

        halo_recv_front_offset = absorb_size;
        halo_send_front_offset = halo_recv_front_offset + halo_size;

        halo_recv_back_offset  = local_grid_size - halo_size;
        halo_send_back_offset  = halo_recv_back_offset - halo_size;
    } else if (last == partition_id){
        sz_local = HALO + local_nz + HALO + ABSORB;
        local_grid_size = sx * sy * sz_local;

        halo_recv_front_offset = 0;
        halo_send_front_offset = halo_recv_front_offset + halo_size;

        halo_recv_back_offset  = local_grid_size - halo_size - absorb_size;
        halo_send_back_offset  = halo_recv_back_offset - halo_size;
    } else {
        sz_local = HALO + local_nz + HALO;
        local_grid_size = sx * sy * sz_local;

        halo_recv_front_offset = 0;
        halo_send_front_offset = halo_recv_front_offset + halo_size;

        halo_recv_back_offset  = local_grid_size - halo_size;
        halo_send_back_offset  = halo_recv_back_offset - halo_size;
    }

    if(is_source_gpu){
        if(total_partitions % 2 == 0) {
            offset = ind(sx/2, sy/2, 0);
        } else {
            offset = ind(sx/2, sy/2, local_nz/2);
        }
    }

}

float Source(float dt, int it){
  float tf, fc, fct, expo;
  tf=TWOSQRTPI/FCUT;
  fc=FCUT/THREESQRTPI;
  fct=fc*(((float)it)*dt-tf);
  expo=PICUBE*fct*fct;
  return ((1.0f-2.0f*expo)*expf(-expo));
}

void backend_init(int rank, int comm_size, int nx, int ny, int nz, int num_gpus){
    int deviceCount, gpu_id;
    int first = 0, last = comm_size - 1;
    partition_id     = rank;
    total_partitions = comm_size;

    compute_grid_dimensions_and_offsets(partition_id, total_partitions, nx, ny, nz);

    cudaGetDeviceCount(&deviceCount);
    gpu_id = partition_id % deviceCount;
    cudaSetDevice(gpu_id);
    cudaStreamCreate(&streams[STREAM_HALO]);
    cudaStreamCreate(&streams[STREAM_COMPUTE]);

    cudaMallocHost(&buffer_out[LEFT],  local_grid_size * sizeof(float));
    cudaMallocHost(&buffer_in[LEFT],   local_grid_size * sizeof(float));

    cudaMallocHost(&buffer_out[RIGHT], local_grid_size * sizeof(float));
    cudaMallocHost(&buffer_in[RIGHT],  local_grid_size * sizeof(float));

    if(rank == 0){
        host_grid = (float*)malloc((sx * sy * sz) * sizeof(float));
    }

    printf("rank %d tamanho %d\n", partition_id, local_grid_size);
}

void backend_data_initialize(){
    int local_grid_size_bytes = local_grid_size * sizeof(float);

    cudaMalloc(&dev_ch1dxx, local_grid_size_bytes);
    cudaMalloc(&dev_ch1dyy, local_grid_size_bytes);
    cudaMalloc(&dev_ch1dzz, local_grid_size_bytes);
    cudaMalloc(&dev_ch1dxy, local_grid_size_bytes);
    cudaMalloc(&dev_ch1dyz, local_grid_size_bytes);
    cudaMalloc(&dev_ch1dxz, local_grid_size_bytes);
    cudaMalloc(&dev_v2px,   local_grid_size_bytes);
    cudaMalloc(&dev_v2pz,   local_grid_size_bytes);
    cudaMalloc(&dev_v2sz,   local_grid_size_bytes);
    cudaMalloc(&dev_v2pn,   local_grid_size_bytes);
    cudaMalloc(&dev_pp,     local_grid_size_bytes);
    cudaMalloc(&dev_pc,     local_grid_size_bytes);
    cudaMalloc(&dev_qp,     local_grid_size_bytes);
    cudaMalloc(&dev_qc,     local_grid_size_bytes);

    cudaMemset(dev_pp, 0, local_grid_size_bytes);
    cudaMemset(dev_pc, 0, local_grid_size_bytes);
    cudaMemset(dev_qp, 0, local_grid_size_bytes);
    cudaMemset(dev_qc, 0, local_grid_size_bytes);
    cudaDeviceSynchronize();

}


void printGrid(){
    int local_grid_size_bytes = local_grid_size * sizeof(float);

    float *temp = (float*) malloc(local_grid_size_bytes);
    cudaMemcpy(temp, dev_pp, local_grid_size_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int z = 0; z < sz_local; z++){
    for(int y = 0; y < sy; y++){
    for(int x = 0; x < sx; x++){
        printf("%0.0f ", temp[(sx * sy * z) + (y * sx) + x]);
    }
    printf("\n");
    }
    printf("\n\n");
    }
    free(temp);
}



void backend_run(int num_steps, int delta_time){

    if(partition_id == 1){
        setHaloWithPartitionSource(partition_id, halo_size,
            halo_recv_front_offset, halo_recv_back_offset, halo_send_front_offset, halo_send_back_offset, dev_pp);
        MPI_Send(dev_pp + halo_send_front_offset, halo_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    if(partition_id == 0){
        MPI_Recv(dev_pp + halo_recv_back_offset, halo_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printGrid();
    }

    float val = 0.0f;
    for(int it = 1; it <= num_steps; it++){
        if(is_source_gpu){
            printf("Rank %d\n -middle insertSource offset %d\n", partition_id, offset);
            val = Source(delta_time, it-1);
            insert_source(val, dev_pc, dev_qc, offset);
        }
    }
}

void backend_finalize(){
    cudaStreamDestroy(streams[STREAM_HALO]);
    cudaStreamDestroy(streams[STREAM_COMPUTE]);
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
