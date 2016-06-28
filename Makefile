
CC = gcc
FLAGS = -std=c99 -Wall -g -Wpedantic -Wno-unknown-pragmas -O3
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
	clover_exchange.o \
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
	accelerate.o


OBJDIR = obj

COBJECTS = $(addprefix $(OBJDIR)/, $(OBJECTS))

CSOURCES = $(OBJECTS:.o=.c)

all: $(COBJECTS) $(FOBJECTS) update_tile_halo_kernel.c Makefile
	$(CC) $(FLAGS) $(COBJECTS) $(FOBJECTS) clover_leaf.c -o clover_leaf

$(OBJDIR)/%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

%.o: %.f90
	gfortran -g -c data.f90 $<

fast: $(CSOURCES) $(FOBJECTS)
	$(CC) $(FLAGS) $(CSOURCES) $(FOBJECTS) clover_leaf.c -o clover_leaf


clean:
	rm -f $(OBJDIR)/* *.o *.mod clover_leaf
	# *.o *.mod *genmod* *cuda* *hmd* *.cu *.oo *.hmf *.lst *.cub *.ptx *.cl clover_leaf

