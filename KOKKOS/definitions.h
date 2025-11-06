#ifndef __KOKKOS_DEFINES
#define __KOKKOS_DEFINES

#include <Kokkos_Core_fwd.hpp>
#include <stdio.h>
#include <Kokkos_Core.hpp>

using HostMemSpace   = Kokkos::HostSpace;

#ifdef KOKKOS_ENABLE_CUDA
  #include <cuda_runtime.h>
  using ExecutionSpace = Kokkos::Cuda;
  using DeviceMemSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
  #include <hip/hip_runtime.h>
  using ExecutionSpace = Kokkos::HIP;
  using DeviceMemSpace = Kokkos::HIPSpace;
#else
  #error "Either KOKKOS_ENABLE_CUDA or KOKKOS_ENABLE_HIP must be defined"
#endif

using DeviceViewFloat1D = Kokkos::View<float*, DeviceMemSpace>;
using HostViewFloat1D   = Kokkos::View<float*, HostMemSpace>;

#endif
