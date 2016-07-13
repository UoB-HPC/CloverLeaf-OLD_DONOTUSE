#include "update_halo.h"
#include "definitions_c.h"
#include "timer_c.h"
#include "update_tile_halo.h"
#include "kernels/update_halo_kernel_c.c"
#include "clover.h"

void update_halo(int *fields, int depth)
{
    double kernel_time = 0.0;
    if (profiler_on) kernel_time = timer();

    update_tile_halo(fields, depth);

    if (profiler_on) {
        profiler.tile_halo_exchange += (timer() - kernel_time);
        kernel_time = timer();
    }

    clover_exchange(fields, depth);

    if (profiler_on) {
        profiler.mpi_halo_exchange = profiler.mpi_halo_exchange + (timer() - kernel_time);
        kernel_time = timer();
    }

    if (chunk.chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE ||
            chunk.chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE ||
            chunk.chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE ||
            chunk.chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
        for (int tile = 0; tile < tiles_per_chunk; tile++)
            update_halo_kernel_c_(
                &chunk.tiles[tile].t_xmin,
                &chunk.tiles[tile].t_xmax,
                &chunk.tiles[tile].t_ymin,
                &chunk.tiles[tile].t_ymax,
                chunk.chunk_neighbours,
                chunk.tiles[tile].tile_neighbours,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.soundspeed,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                chunk.tiles[tile].field.vol_flux_x,
                chunk.tiles[tile].field.vol_flux_y,
                chunk.tiles[tile].field.mass_flux_x,
                chunk.tiles[tile].field.mass_flux_y,
                fields,
                &depth);
    }

}
