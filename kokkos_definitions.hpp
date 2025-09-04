#pragma once

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
  #include <cuda_runtime.h>
  using streamType = cudaStream_t;
  using ExecSpaceType = Kokkos::Cuda;

  // Função auxiliar para criar stream
  inline void create_stream(int device_id, streamType& stream) {
      cudaSetDevice(device_id);
      cudaStreamCreate(&stream);
  }

  // Função auxiliar para destruir stream
  inline void destroy_stream(streamType& stream) {
      cudaStreamDestroy(stream);
  }

#elif defined(KOKKOS_ENABLE_HIP)
  #include <hip/hip_runtime.h>
  // Kokkos::HIP agora é o padrão
  using streamType = hipStream_t;
  using ExecSpaceType = Kokkos::HIP;

  // Função auxiliar para criar stream
  inline void create_stream(int device_id, streamType& stream) {
      hipSetDevice(device_id);
      hipStreamCreate(&stream);
  }

  // Função auxiliar para destruir stream
  inline void destroy_stream(streamType& stream) {
      hipStreamDestroy(stream);
  }

#else
  #error "Either KOKKOS_ENABLE_CUDA or KOKKOS_ENABLE_HIP must be defined"
#endif

using HostMemSpace   = Kokkos::HostSpace;
// É mais seguro usar o memory_space do Execution Space que você definiu
using DeviceMemSpace = ExecSpaceType::memory_space;
