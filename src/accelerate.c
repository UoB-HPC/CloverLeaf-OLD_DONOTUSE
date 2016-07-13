#include "accelerate.h"
#include "definitions_c.h"
#include "kernels/accelerate_kernel_c.c"
#include "timer_c.h"

void accelerate()
{
    double kernel_time = 0.0;
    if (profiler_on) kernel_time = timer();

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax + 1,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax + 1,
            ({
                accelerate_kernel_c_(j, k,
                chunk.tiles[tile].t_xmin, chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin, chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.xarea,
                chunk.tiles[tile].field.yarea,
                chunk.tiles[tile].field.volume,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                dt);
            }));
        }
#ifdef USE_KOKKOS
        Kokkos::fence();
#endif
    }

    if (profiler_on) profiler.acceleration += timer() - kernel_time;
}

