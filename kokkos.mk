# ========= Compiladores e Caminhos =========
# O KOKKOS_PATH deve apontar para a biblioteca KOKKOS, não para o wrapper. (CORRIGIDO)
KOKKOS_PATH       = $(shell spack location -i kokkos)
NVCC_WRAPPER_PATH = $(shell spack location -i kokkos-nvcc-wrapper)

CC            = $(NVCC_WRAPPER_PATH)/bin/nvcc_wrapper
# INCLUDES e LIBS agora usam o KOKKOS_PATH correto. (CORRIGIDO)
INCLUDES      = -I$(KOKKOS_PATH)/include
LIBS          = -L$(KOKKOS_PATH)/lib -lkokkoscore -lkokkoscontainers -lkokkossimd
CXXFLAGS      = -std=c++17 -O3
CXXFLAGS      += -DBACKEND_KOKKOS
LDFLAGS       =

# ========= Extração Explícita de Flags MPI =========
# --- Para Open MPI ---
MPI_CXXFLAGS  = $(shell mpic++ --showme:compile)
MPI_LIBS      = $(shell mpic++ --showme:link)

# --- Para MPICH e derivados (MVAPICH, Intel MPI) ---
# MPI_CXXFLAGS  = $(shell mpic++ -compile_info)
# MPI_LIBS      = $(shell mpic++ -link_info)


# ========= Arquivos do Projeto =========
SRCS          = main_kokkos.cpp backend_kokkos.cpp
OUTPUT        = fletcher.x
export KOKKOS_CXX_COMPILER := $(shell which mpic++)


# ========= Regras =========
all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(CC) $(CXXFLAGS) $(INCLUDES) $(MPI_CXXFLAGS) $(SRCS) $(LIBS) $(MPI_LIBS) $(LDFLAGS) -o $@

clean:
	rm -f $(OUTPUT) *.o
