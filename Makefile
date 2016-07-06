
CC_INTEL = mpiicpc
CC_GNU = mpic++
CC_ = mpic++

FLAGS_GNU = -std=c++11 -Wall -Wpedantic -g -Wno-unknown-pragmas -O3 -march=native -lm
FLAGS_INTEL = -std=c++11 -O3 -g -fp-model strict -march=native
FLAGS_ = 

FLAGS = $(FLAGS_$(COMPILER))
CC = $(CC_$(COMPILER))

OBJECTS = data_c.o \
	definitions_c.o \
	initialise.o \
	hydro.o \
	report.o \
	start.o \
	build_field.o \
	initialise_chunk.o \
	generate_chunk.o \
	ideal_gas.o \
	update_halo.o \
	update_tile_halo.o \
	field_summary.o \
	timer_c.o \
	timestep.o \
	viscosity.o \
	calc_dt.o \
	PdV.o \
	revert.o \
	flux_calc.o \
	advection.o \
	reset_field.o \
	visit.o \
	accelerate.o \
	clover.o



default: build

CXX = mpic++
# CXX = 
# KOKKOS_PATH=/usr/local/lib/kokkos
# KOKKOS_PATH=/Users/jamus/kokkos-tutorial/kokkos

include $(KOKKOS_PATH)/Makefile.kokkos

OBJDIR = obj
SRCDIR = src

COBJECTS = $(addprefix $(OBJDIR)/, $(OBJECTS))
CSOURCES = $(addprefix $(SRCDIR)/, $(OBJECTS:.o=.c))

build: $(COBJECTS) Makefile $(KOKKOS_LINK_DEPENDS) $(KERNELS)
	$(CC) $(KOKKOS_LDFLAGS) $(FLAGS) $(KOKKOS_CPPFLAGS) $(EXTRA_PATH) $(COBJECTS) $(KOKKOS_LIBS) $(SRCDIR)/clover_leaf.c -o clover_leaf

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(KOKKOS_CPP_DEPENDS)
	$(CC) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(FLAGS) $(EXTRA_INC) -c $< -o $@


fast: $(CSOURCES)
	$(CC) $(FLAGS) $(CSOURCES) $(SRCDIR)/clover_leaf.c -o clover_leaf

clean: kokkos-clean
	rm -f $(OBJDIR)/* *.o clover_leaf

print-%  : ; @echo $* = $($*)