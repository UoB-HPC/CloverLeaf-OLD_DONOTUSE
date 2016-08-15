#include "flux_calc.h"
// #include "kernels/flux_calc_kernel_c.c"
#include "adaptors/flux_calc.cpp"
#include "definitions_c.h"
#include "timer_c.h"


void flux_calc()
{
    double kernel_time = 0.0;

    if (profiler_on) kernel_time = timer();

    flux_calc(chunk, dt);

    if (profiler_on) profiler.flux += timer() - kernel_time;
}
