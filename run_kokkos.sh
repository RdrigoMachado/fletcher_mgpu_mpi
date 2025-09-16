source ~/tupi/spack/share/spack/setup-env.sh
KOKKOS_LIB_PATH=$(spack location -i kokkos)/lib
#export LD_LIBRARY_PATH="$(spack location -i kokkos)/lib:$LD_LIBRARY_PATH"
mpirun -H tupi4:1,tupi5:1 -np 2 -x LD_LIBRARY_PATH=$KOKKOS_LIB_PATH ./fletcher.x 180 180 180 100 2
