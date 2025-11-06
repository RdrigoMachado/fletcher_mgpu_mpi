#include <cuda_runtime.h>
#include "backend.hpp"
void insert_source(const float val, float *qp, float *qc, int offset);
void setHaloWithPartitionSource(int partition_id, int halo_size,
    int halo_recv_front_offset, int halo_recv_back_offset, int halo_send_front_offset, int halo_send_back_offset,
    float *pp);
