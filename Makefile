# ========= Configurações =========

# Compiladores principais
MPICXX     = mpic++
NVCC       = nvcc

OUTPUT     = fletcher.x

# Flags comuns
CXXFLAGS   = -std=c++17 -O3
INCLUDES   = -I/usr/local/cuda/include
LIBS       = -L/usr/local/cuda/lib64
LDFLAGS    = -lcudart -lcuda

# Fontes
SRCS_CPP   = main.cpp backend_cuda.cpp
SRCS_CU    = kernel_cuda.cu

# Objetos
OBJS_CPP   = $(SRCS_CPP:.cpp=.o)
OBJS_CU    = $(SRCS_CU:.cu=.o)
OBJS       = $(OBJS_CPP) $(OBJS_CU)

all: $(OUTPUT)

# Regra final de link: usa mpic++
$(OUTPUT): $(OBJS)
	$(MPICXX) $(CXXFLAGS) $(OBJS) $(LIBS) $(LDFLAGS) -o $@

# Compilar arquivos .cpp com mpic++
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilar arquivos .cu com nvcc
%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OUTPUT) *.o
