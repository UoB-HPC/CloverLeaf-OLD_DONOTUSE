#include "accelerate.h"
#include "definitions_c.h"
#include "accelerate_kernel_c.c"
#include "timer_c.h"

void accelerate()
{
    double kernel_time = 0.0;
    if (profiler_on) kernel_time = timer();

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        accelerate_kernel_c_(&chunk.tiles[tile], dt);
    }

    if (profiler_on) profiler.acceleration += timer() - kernel_time;
}

