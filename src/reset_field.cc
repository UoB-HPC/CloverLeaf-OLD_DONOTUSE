#include "reset_field.h"
#include "definitions_c.h"
#include "adaptors/reset_field.cpp"
#include "timer_c.h"


void reset_field()
{
    double kernel_time = 0.0;
    if (profiler_on) kernel_time = timer();

    reset_field(chunk);

    if (profiler_on) profiler.reset += timer() - kernel_time;
}
