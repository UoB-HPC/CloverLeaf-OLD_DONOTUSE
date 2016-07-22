#include "accelerate.h"
#include "definitions_c.h"
#include "adaptors/accelerate.c"
#include "timer_c.h"

void accelerate_kokkos();
void accelerate_openmp();

void accelerate()
{
    double kernel_time = 0.0;
    if (profiler_on) {kernel_time = timer();}
    accelerate_adaptor();
    if (profiler_on) {profiler.acceleration += timer() - kernel_time;}
}
