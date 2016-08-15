#include "PdV.h"
#include "definitions_c.h"
#include "timer_c.h"
#include "ideal_gas.h"
#include "adaptors/pdv.cpp"
#include "revert.h"
#include "update_halo.h"

void PdV(bool predict)
{
    // error_condition = 0;
    double kernel_time = 0.0;

    if (profiler_on) kernel_time = timer();

    pdv(chunk, predict, dt);

    if (profiler_on) profiler.PdV += timer() - kernel_time;

    if (predict) {
        if (profiler_on) kernel_time = timer();
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            ideal_gas(tile, true);
        }
        if (profiler_on) profiler.ideal_gas += timer() - kernel_time;

        int fields[NUM_FIELDS];
        for (int i = 0; i < NUM_FIELDS; i++) {
            fields[i] = 0;
        }
        fields[FIELD_PRESSURE] = 1;
        update_halo(fields, 1);
    }

    if (predict) {
        if (profiler_on) kernel_time = timer();
        revert();
        if (profiler_on) profiler.revert += timer() - kernel_time;
    }
}
