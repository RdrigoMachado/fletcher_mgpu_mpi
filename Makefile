# ========= Configurações =========

# Backends disponíveis: CUDA, KOKKOS, SYCL
BACKEND ?= CUDA

# Compiladores principais
NVCC        = nvcc
NVCC_WRAPPER= nvcc_wrapper
MPI_WRAPP   = mpic++
DPCPP       = dpcpp

OUTPUT    = fletcher.x

# Flags comuns
CXXFLAGS  = -std=c++17 -O3
LDFLAGS   =
INCLUDES  =
LIBS      =
SRCS      = main.cpp

# ========= Seleção de Backend =========
ifeq ($(BACKEND), CUDA)
    CC       = $(NVCC)
    SRCS    += backend_cuda.cpp
    INCLUDES+= -I/usr/local/cuda/include
    LIBS    += -L/usr/local/cuda/lib64
    LDFLAGS += -lcudart -lcuda
    # usa mpic++ como compilador host
    HOSTCOMP = -ccbin=$(MPI_WRAPP)
endif

ifeq ($(BACKEND), KOKKOS)
	# Encontra a instalação do Kokkos via Spack
	KOKKOS_PATH = $(shell spack location -i kokkos-nvcc-wrapper)
	CC          = $(KOKKOS_PATH)/bin/nvcc_wrapper
	SRCS        += backend_kokkos.cpp

	# A MÁGICA ACONTECE AQUI! ✨
	# Dizemos ao nvcc_wrapper para usar o mpic++ como compilador C++ subjacente.
	# O mpic++ então encontrará o mpi.h e as bibliotecas MPI automaticamente.
	# Colocamos o "export" diretamente na regra de compilação para garantir que
	# o ambiente seja configurado corretamente para o comando.
	export KOKKOS_CXX_COMPILER=/usr/bin/mpic++ &&
    echo $KOKKOS_CXX_COMPILER
	# NÃO PRECISA DESTAS LINHAS! O wrapper cuida de tudo.
	# INCLUDES+= -I$(KOKKOS_PATH)/include
	# LIBS      += -L$(KOKKOS_PATH)/lib -lkokkoscore ...
	# LDFLAGS += -Wl,-rpath,$(KOKKOS_PATH)/lib
endif

ifeq ($(BACKEND), SYCL)
    CC       = $(DPCPP)
    SRCS    += backend_sycl.cpp
    # Ajuste conforme a instalação do SYCL
    INCLUDES+= -I/opt/sycl/include
    LIBS    += -L/opt/sycl/lib
    LDFLAGS += -lsycl
endif

# ========= Regras =========
all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(KOKKOS_EXPORT) $(CC) $(CXXFLAGS) $(INCLUDES) $(SRCS) $(LIBS) $(LDFLAGS) $(HOSTCOMP) -o $@

clean:
	rm -f $(OUTPUT) *.o
