
#include "definitions_c.h"
#include <stdio.h>
#include "report.h"
void read_config(FILE* in);
void initialise()
{
    int ios,
        get_unit,
        stat;
    FILE *out_unit,
         *uin;

    g_out = fopen("clover.out", "w");
    if (g_out == NULL) {
        report_error("initialise", "Error opening clover.out file.");
    }

    fprintf(g_out, "Clover Version %8.3f", g_version);
    fprintf(stdout, "Output file clover.out opened. All output will go there.\n");

    fprintf(g_out, "Clover will run from the following input:-\n");

    uin = fopen("clover.in", "r");
    if (uin == NULL) {
        out_unit = fopen("clover.in", "w");
        fprintf(out_unit, "*clover");
        fprintf(out_unit, " state 1 density=0.2 energy=1.0");
        fprintf(out_unit, " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0");
        fprintf(out_unit, " x_cells=10");
        fprintf(out_unit, " y_cells=2");
        fprintf(out_unit, " xmin=0.0");
        fprintf(out_unit, " ymin=0.0");
        fprintf(out_unit, " xmax=10.0");
        fprintf(out_unit, " ymax=2.0");
        fprintf(out_unit, " initial_timestep=0.04");
        fprintf(out_unit, " timestep_rise=1.5");
        fprintf(out_unit, " max_timestep=0.04");
        fprintf(out_unit, " end_time=3.0");
        fprintf(out_unit, " test_problem 1");
        fprintf(out_unit, "*endclover");
        uin = fopen("clover.in", "r");
    }
    read_config(uin);
}

void read_config(FILE* in)
{

    test_problem = 0;

    int state_max = 0;

    grid.xmin =  0.0;
    grid.ymin =  0.0;
    grid.xmax = 100.0;
    grid.ymax = 100.0;

    grid.x_cells = 10;
    grid.y_cells = 10;

    end_time = 10.0;
    end_step = g_ibig;
    complete = false;

    visit_frequency = 0;
    summary_frequency = 10;

    tiles_per_chunk = 1;

    dtinit = 0.1;
    dtmax = 1.0;
    dtmin = 0.0000001;
    dtrise = 1.5;
    dtc_safe = 0.7;
    dtu_safe = 0.5;
    dtv_safe = 0.5;
    dtdiv_safe = 0.7;

    use_fortran_kernels = false;
    use_C_kernels = true;
    use_OA_kernels = false;
    profiler_on = false;
    profiler.timestep = 0.0;
    profiler.acceleration = 0.0;
    profiler.PdV = 0.0;
    profiler.cell_advection = 0.0;
    profiler.mom_advection = 0.0;
    profiler.viscosity = 0.0;
    profiler.ideal_gas = 0.0;
    profiler.visit = 0.0;
    profiler.summary = 0.0;
    profiler.reset = 0.0;
    profiler.revert = 0.0;
    profiler.flux = 0.0;
    profiler.tile_halo_exchange = 0.0;
    profiler.self_halo_exchange = 0.0;
    profiler.mpi_halo_exchange = 0.0;

    fprintf(g_out, "Reading input file\n");

    char *line = NULL;
    ssize_t n;
    size_t len = 0;

    while ((n = getline(&line, &len, g_out)) != -1) {
        printf("Retrieved line of length %zu :\n", n);
        printf("%s", line);
    }
    // states = malloc(sizeof(struct state_type) )
}