
CC = gcc-6
FLAGS = -std=c99 -march=native -Wall -g -Wpedantic -Wno-unknown-pragmas -O3
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
SRCDIR = src

COBJECTS = $(addprefix $(OBJDIR)/, $(OBJECTS))
CSOURCES = $(addprefix $(SRCDIR)/, $(OBJECTS:.o=.c))

all: $(COBJECTS) $(FOBJECTS) Makefile
	$(CC) $(FLAGS) $(COBJECTS) $(FOBJECTS) $(SRCDIR)/clover_leaf.c -o clover_leaf

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(FLAGS) -c $< -o $@

fast: $(CSOURCES) $(FOBJECTS)
	$(CC) $(FLAGS) $(CSOURCES) $(FOBJECTS) $(SRCDIR)/clover_leaf.c -o clover_leaf

clean:
	rm -f $(OBJDIR)/* *.o clover_leaf
