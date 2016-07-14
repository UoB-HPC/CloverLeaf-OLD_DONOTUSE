#include "reset_field.h"
#include "definitions_c.h"
#include "kernels/reset_field_kernel_c.c"
#include "timer_c.h"


void reset_field()
{
    double kernel_time = 0.0;
    if (profiler_on) kernel_time = timer();

    for (int tile = 0; tile < tiles_per_chunk; tile++) {

        #pragma omp parallel
        {
            reset_field_kernel_c_(
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.yvel1);

        }
    }

    if (profiler_on) profiler.reset += timer() - kernel_time;
}
