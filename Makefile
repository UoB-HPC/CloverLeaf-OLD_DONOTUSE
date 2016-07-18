
MPI_CC_INTEL = mpiicpc
MPI_CC_GNU = mpic++
CC_CUDA = nvcc_wrapper
MPI_CC_ = mpic++


FLAGS_GNU = -std=c++11 -Wall -Wpedantic -g -Wno-unknown-pragmas -O3 -march=native -lm
FLAGS_INTEL = -std=c++11 -O3 -g -restrict -march=native -fp-model strict
# FLAGS_INTEL = -std=c++11 -O3 -g -restrict -march=native -no-prec-div -fno-alias
FLAGS_CUDA = 
FLAGS_ = 

MPI_FLAGS = $(FLAGS_$(COMPILER))
MPI_CC = $(MPI_CC_$(COMPILER))

OBJECTS = data_c.o \
	definitions_c.o \
	report.o \
	initialise_chunk.o \
	generate_chunk.o \
	ideal_gas.o \
	timer_c.o \
	viscosity.o \
	calc_dt.o \
	PdV.o \
	revert.o \
	flux_calc.o \
	advection.o \
	reset_field.o \
	visit.o \
	accelerate.o \
	field_summary_driver.o

	# update_tile_halo.o 
MPIOBJECTS = clover.o \
	initialise.o \
	hydro.o \
	update_halo.o \
	timestep.o \
	start.o \
	field_summary.o



default: build

ifdef USE_KOKKOS

ifeq ($(USE_KOKKOS),gpu)

CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
MPI_FLAGS += -lcudart
FLAGS += -O3

KOKKOS_CUDA_OPTIONS = "enable_lambda"
KOKKOS_DEVICE= "Cuda"

else
CXX = $(MPI_CC)
FLAGS += $(MPI_FLAGS)
endif

MPI_FLAGS += -DUSE_KOKKOS
FLAGS += -DUSE_KOKKOS
CC = $(CXX)
include $(KOKKOS_PATH)/Makefile.kokkos

else

CC = $(MPI_CC)

FLAGS += -fopenmp $(MPI_FLAGS)
MPI_FLAGS += -fopenmp

endif

OBJDIR    = obj
MPIOBJDIR = mpiobj
SRCDIR    = src

_OBJECTS = $(addprefix $(OBJDIR)/, $(OBJECTS))
_SOURCES = $(addprefix $(SRCDIR)/, $(OBJECTS:.o=.cpp))
_MPIOBJECTS = $(addprefix $(MPIOBJDIR)/, $(MPIOBJECTS))
_MPISOURCES = $(addprefix $(SRCDIR)/, $(MPIOBJECTS:.o=.cpp))

-include $(_OBJECTS:.o=.d)

build: $(_OBJECTS) $(_MPIOBJECTS) Makefile $(KOKKOS_LINK_DEPENDS) $(KERNELS)
	$(MPI_CC) $(MPI_FLAGS) $(KOKKOS_CPPFLAGS) $(EXTRA_PATH) $(_OBJECTS) $(_MPIOBJECTS) $(SRCDIR)/clover_leaf.cpp $(KOKKOS_LIBS) $(KOKKOS_LDFLAGS) -o clover_leaf

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CC) $(FLAGS) $(MPIINCLUDE) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(EXTRA_INC) -c $< -o $@
	# $(CC) $(FLAGS) $(MPIINCLUDE) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(EXTRA_INC) -MM $< > $(OBJDIR)/$*.d

$(MPIOBJDIR)/%.o: $(SRCDIR)/%.cpp $(KOKKOS_CPP_DEPENDS)
	$(MPI_CC) $(MPI_FLAGS) $(KOKKOS_CPPFLAGS) $(EXTRA_INC) -c $< -o $@
	# $(MPI_CC) $(MPI_FLAGS) $(KOKKOS_CPPFLAGS) $(EXTRA_INC) -MM $< > $(OBJDIR)/$*.d


fast: $(_SOURCES) $(_MPISOURCES) Makefile $(KOKKOS_LINK_DEPENDS) $(KERNELS)
	$(MPI_CC) $(MPI_FLAGS) $(KOKKOS_CPPFLAGS) $(EXTRA_PATH) $(_SOURCES) $(_MPISOURCES) $(SRCDIR)/clover_leaf.c $(KOKKOS_LIBS) $(KOKKOS_LDFLAGS) -o clover_leaf

clean: 
	rm -f $(OBJDIR)/* $(MPIOBJDIR)/* *.o clover_leaf

print-%  : ; @echo $* = $($*)