#include "backend.hpp"

sycl::queue queues[2];
sycl::device device;
float *buffer_out[2];
float *buffer_in[2];
int sxsysz;
int sxsy, sz_slice;

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
    int num_boards  = 0;
    process_num     = rank;
    universe_size   = comm_size;
    sxsy            = sx * sy;
    sz_slice        = sz / universe_size;

    if(0 != rank){
        num_boards++;
        buffer_out[LEFT]  = sycl::malloc_host<float>(sxsysz, queues[HALO]);
        buffer_in[LEFT]   = sycl::malloc_host<float>(sxsysz, queues[HALO]);
    }
    if (comm_size - 1 != rank){
        num_boards++;
        buffer_out[RIGHT] = sycl::malloc_host<float>(sxsysz, queues[HALO]);
        buffer_in[RIGHT]  = sycl::malloc_host<float>(sxsysz, queues[HALO]);
    }

    sxsysz = sx * sy * sz + (num_boards * BOARD_SIZE);

    std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    int gpu_id      = rank % devices.size();
    device          = devices[gpu_id];
    queues[HALO]    = sycl::queue(device);
    queues[COMPUTE] = sycl::queue(device);
}

void backend_data_initialize(){
    dev_ch1dxx = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_ch1dyy = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_ch1dzz = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_ch1dxy = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_ch1dyz = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_ch1dxz = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_v2px   = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_v2pz   = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_v2sz   = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_v2pn   = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_pp     = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_pc     = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_qp     = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    dev_qc     = sycl::malloc_device<float>(sxsysz, queues[HALO]);
    sxsysz     = sycl::malloc_device<float>(sxsysz, queues[HALO]);
}

void backend_run(int num_steps){
    insert_source();
    swap_borders(int universe_size, int process_num, );
}

void backend_finalize(){
     sycl::free(dev_ch1dxx, queues_halo[HALO]);
     sycl::free(dev_ch1dyy, queues_halo[HALO]);
     sycl::free(dev_ch1dzz, queues_halo[HALO]);
     sycl::free(dev_ch1dxy, queues_halo[HALO]);
     sycl::free(dev_ch1dyz, queues_halo[HALO]);
     sycl::free(dev_ch1dxz, queues_halo[HALO]);
     sycl::free(dev_v2px,   queues_halo[HALO]);
     sycl::free(dev_v2pz,   queues_halo[HALO]);
     sycl::free(dev_v2sz,   queues_halo[HALO]);
     sycl::free(dev_v2pn,   queues_halo[HALO]);
     sycl::free(dev_pp,     queues_halo[HALO]);
     sycl::free(dev_pc,     queues_halo[HALO]);
     sycl::free(dev_qp,     queues_halo[HALO]);
     sycl::free(dev_qc,     queues_halo[HALO]);
     sycl::free(sxsysz,     queues_halo[HALO]);
}
