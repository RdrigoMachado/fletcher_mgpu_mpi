#include "backend.hpp"
#include "kokkos_definitions.hpp"

std::unique_ptr<Backend_Data> pImpl;
int sxsysz;
int slice_number;

void backend_init(int rank, int comm_size, int sx, int sy, int sz){
    int num_boards = 1;
    slice_number = rank;
    if(0 != rank && comm_size - 1 != rank)
        num_boards = 2;
    sxsysz = sx * sy * sz + (num_boards * BOARD_SIZE);

    pImpl = std::make_unique<Backend_Data>();
    #ifdef KOKKOS_ENABLE_CUDA
        streamType stream_halo, stream_compute;
        int deviceCount, gpu_id;
        cudaGetDeviceCount(&deviceCount);
        gpu_id = rank % deviceCount;
        cudaSetDevice(gpu_id);
        // cudaStreamCreate(&stream_halo);
        // cudaStreamCreate(&stream_compute);
        // pImpl->exec_spaces[HALO] = ExecSpaceType(stream_halo);
        // pImpl->exec_spaces[COMPUTE] = ExecSpaceType(stream_compute);
    #elif defined(KOKKOS_ENABLE_HIP)
        streamType stream_halo, stream_compute;
        int deviceCount, gpu_id;
        hipGetDeviceCount(&deviceCount);
        gpu_id = rank % deviceCount;
        hipSetDevice(gpu_id);
        // hipStreamCreate(&stream_halo);
        // hipStreamCreate(&stream_compute);
        // pImpl->exec_spaces[HALO] = ExecSpaceType(stream_halo);
        // pImpl->exec_spaces[COMPUTE] = ExecSpaceType(stream_compute);
    #endif

}

void backend_data_initialize(){
    pImpl->dev_ch1dxx = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_ch1dyy = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_ch1dzz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_ch1dxy = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_ch1dyz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_ch1dxz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_v2px = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_v2pz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_v2sz = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_v2pn = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_pp = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_pc = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_qp = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
    pImpl->dev_qc = Kokkos::View<float*, DeviceMemSpace>(Kokkos::view_alloc(exec_spaces[HALO]), sxsysz);
}

void backend_run(int num_steps){

}

void backend_finalize(){
    #ifdef KOKKOS_ENABLE_CUDA
        cudaStreamDestroy(pImpl->exec_spaces[HALO].cuda_stream());
        cudaStreamDestroy(pImpl->exec_spaces[COMPUTE].cuda_stream());
    #elif defined(KOKKOS_ENABLE_HIP)
        hipStreamDestroy(pImpl->exec_spaces[HALO].hip_stream());
        hipStreamDestroy(pImpl->exec_spaces[COMPUTE].hip_stream());
    #endif
}
