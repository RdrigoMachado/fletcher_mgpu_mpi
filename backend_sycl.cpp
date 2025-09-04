#include "backend.hpp"

sycl::queue queues[2];

void backend_init(int gpu_id){
    std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::device device = devices[gpu_id];
    queues[HALO]    = sycl::queue(device);
    queues[COMPUTE] = sycl::queue(device);
}

void backend_data_initialize(){

}

void backend_run(int num_steps){

}

void backend_finalize(){

}
