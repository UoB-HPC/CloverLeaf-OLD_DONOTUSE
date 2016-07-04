// #include "advection.h"
#include "definitions_c.h"
#include "advec_cell_kernel_c.c"
#include "update_halo.h"
#include "advec_mom_kernel_c.c"
#include "timer_c.h"


void advec_cell_driver(int tile, int sweep_number, int dir);
void advec_mom_driver(int tile, int which_vel, int direction, int sweep_number);

void advection()
{
    int sweep_number, direction;
    int xvel, yvel;
    int fields[NUM_FIELDS];
    double kernel_time = 0.0;

    sweep_number = 1;
    if (advect_x)
        direction = g_xdir;
    else
        direction = g_ydir;

    xvel = g_xdir;
    yvel = g_ydir;

    for (int i = 0; i < NUM_FIELDS; i++) {
        fields[i] = 0;
    }
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_DENSITY1] = 1;
    fields[FIELD_VOL_FLUX_X] = 1;
    fields[FIELD_VOL_FLUX_Y] = 1;
    update_halo(fields, 2);

    if (profiler_on) kernel_time = timer();
    for (int tile  = 0; tile < tiles_per_chunk; tile++) {
        advec_cell_driver(tile, sweep_number, direction);
    }

    if (profiler_on) profiler.cell_advection += timer() - kernel_time;

    for (int i = 0; i < NUM_FIELDS; i++) {
        fields[i] = 0;
    }
    fields[FIELD_DENSITY1] = 1;
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_XVEL1] = 1;
    fields[FIELD_YVEL1] = 1;
    fields[FIELD_MASS_FLUX_X] = 1;
    fields[FIELD_MASS_FLUX_Y] = 1;
    update_halo(fields, 2);

    if (profiler_on) kernel_time = timer();
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        advec_mom_driver(tile, xvel, direction, sweep_number);
        advec_mom_driver(tile, yvel, direction, sweep_number);
    }
    if (profiler_on) profiler.mom_advection += timer() - kernel_time;

    sweep_number = 2;

    if (advect_x) direction = g_ydir;
    if (!advect_x) direction = g_xdir;

    if (profiler_on) kernel_time = timer();
    for (int tile  = 0; tile < tiles_per_chunk; tile++) {
        advec_cell_driver(tile, sweep_number, direction);
    }
    if (profiler_on) profiler.cell_advection = timer() - kernel_time;

    for (int i = 0; i < NUM_FIELDS; i++) {
        fields[i] = 0;
    }
    fields[FIELD_DENSITY1] = 1;
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_XVEL1] = 1;
    fields[FIELD_YVEL1] = 1;
    fields[FIELD_MASS_FLUX_X] = 1;
    fields[FIELD_MASS_FLUX_Y] = 1;
    update_halo(fields, 2);

    if (profiler_on) kernel_time = timer();
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        advec_mom_driver(tile, xvel, direction, sweep_number);
        advec_mom_driver(tile, yvel, direction, sweep_number);
    }
    if (profiler_on) profiler.mom_advection += timer() - kernel_time;
}


void advec_cell_driver(int tile, int sweep_number, int dir)
{
    advec_cell_kernel_c_(
        &chunk.tiles[tile],
        dir,
        sweep_number);
}

void advec_mom_driver(int tile, int which_vel, int direction, int sweep_number)
{
    advec_mom_kernel_c_(
        &chunk.tiles[tile],
        which_vel == 1 ? chunk.tiles[tile].field.xvel1 : chunk.tiles[tile].field.yvel1,
        sweep_number,
        direction);
}
