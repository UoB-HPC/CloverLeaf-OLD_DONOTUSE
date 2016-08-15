#include "update_halo.h"
#include "definitions_c.h"
#include "timer_c.h"
#include "adaptors/update_tile_halo.cpp"
// #include "kernels/update_halo_kernel_c.c"
#include "adaptors/update_local_halo.cpp"
#include "clover.h"

void update_halo(int* fields, int depth)
{
    double kernel_time = 0.0;
    if (profiler_on) kernel_time = timer();

    // update_tile_halo(fields, depth);

    if (profiler_on) {
        profiler.tile_halo_exchange += (timer() - kernel_time);
        kernel_time = timer();
    }

    // clover_exchange(fields, depth);

    if (profiler_on) {
        profiler.mpi_halo_exchange = profiler.mpi_halo_exchange + (timer() - kernel_time);
        kernel_time = timer();
    }

    if (chunk.chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE ||
            chunk.chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE ||
            chunk.chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE ||
            chunk.chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            update_local_halo(chunk.tiles[tile], chunk.chunk_neighbours, fields, depth);
        }
    }
    if (profiler_on) {
        profiler.self_halo_exchange += (timer() - kernel_time);
        kernel_time = timer();
    }
}
