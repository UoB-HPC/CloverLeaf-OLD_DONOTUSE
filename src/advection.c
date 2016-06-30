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
        &chunk.tiles[tile].t_xmin,
        &chunk.tiles[tile].t_xmax,
        &chunk.tiles[tile].t_ymin,
        &chunk.tiles[tile].t_ymax,
        &dir,
        &sweep_number,
        chunk.tiles[tile].field.vertexdx,
        chunk.tiles[tile].field.vertexdy,
        chunk.tiles[tile].field.volume,
        chunk.tiles[tile].field.density1,
        chunk.tiles[tile].field.energy1,
        chunk.tiles[tile].field.mass_flux_x,
        chunk.tiles[tile].field.vol_flux_x,
        chunk.tiles[tile].field.mass_flux_y,
        chunk.tiles[tile].field.vol_flux_y,
        chunk.tiles[tile].field.work_array1,
        chunk.tiles[tile].field.work_array2,
        chunk.tiles[tile].field.work_array3,
        chunk.tiles[tile].field.work_array4,
        chunk.tiles[tile].field.work_array5,
        chunk.tiles[tile].field.work_array6,
        chunk.tiles[tile].field.work_array7);
}

void advec_mom_driver(int tile, int which_vel, int direction, int sweep_number)
{
    if (which_vel == 1) {
        advec_mom_kernel_c_(
            &chunk.tiles[tile].t_xmin,
            &chunk.tiles[tile].t_xmax,
            &chunk.tiles[tile].t_ymin,
            &chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.xvel1,
            chunk.tiles[tile].field.mass_flux_x,
            chunk.tiles[tile].field.vol_flux_x,
            chunk.tiles[tile].field.mass_flux_y,
            chunk.tiles[tile].field.vol_flux_y,
            chunk.tiles[tile].field.volume,
            chunk.tiles[tile].field.density1,
            chunk.tiles[tile].field.work_array1,
            chunk.tiles[tile].field.work_array2,
            chunk.tiles[tile].field.work_array3,
            chunk.tiles[tile].field.work_array4,
            chunk.tiles[tile].field.work_array5,
            chunk.tiles[tile].field.work_array6,
            chunk.tiles[tile].field.celldx,
            chunk.tiles[tile].field.celldy,
            &which_vel,
            &sweep_number,
            &direction);
    } else {
        advec_mom_kernel_c_(
            &chunk.tiles[tile].t_xmin,
            &chunk.tiles[tile].t_xmax,
            &chunk.tiles[tile].t_ymin,
            &chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.yvel1,
            chunk.tiles[tile].field.mass_flux_x,
            chunk.tiles[tile].field.vol_flux_x,
            chunk.tiles[tile].field.mass_flux_y,
            chunk.tiles[tile].field.vol_flux_y,
            chunk.tiles[tile].field.volume,
            chunk.tiles[tile].field.density1,
            chunk.tiles[tile].field.work_array1,
            chunk.tiles[tile].field.work_array2,
            chunk.tiles[tile].field.work_array3,
            chunk.tiles[tile].field.work_array4,
            chunk.tiles[tile].field.work_array5,
            chunk.tiles[tile].field.work_array6,
            chunk.tiles[tile].field.celldx,
            chunk.tiles[tile].field.celldy,
            &which_vel,
            &sweep_number,
            &direction);
    }
}
