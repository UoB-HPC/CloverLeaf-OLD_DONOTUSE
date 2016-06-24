#include "hydro.h"
#include "definitions_c.h"
#include "timer_c.h"
#include <stdbool.h>
#include "timestep.h"
#include "PdV.h"
#include "flux_calc.h"
#include "advection.h"
#include "reset_field.h"
#include "visit.h"
#include "field_summary.h"
#include <xmmintrin.h>
#include "accelerate.h"


void hydro()
{
    double timerstart = timer(),
           step_time,
           first_step, second_step,
           wall_clock;

    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

    while (true) {
        step_time = timer();
        step++;

        timestep();

        PdV(true);

        accelerate();

        PdV(false);

        flux_calc();

        advection();

        reset_field();

        advect_x = !advect_x;

        _time += dt;

        if (summary_frequency != 0) {
            if (step % summary_frequency == 0)
                field_summary();
        }
        if (visit_frequency != 0) {
            if (step % visit_frequency == 0) visit();
        }

        if (step == 1) first_step = timer() - step_time;
        if (step == 2) second_step = timer() - step_time;

        if ((_time + g_small) > end_time || step >= end_step) {
            complete = true;
            field_summary();
            if (visit_frequency != 0) visit();

            wall_clock = timer() - timerstart;

            fprintf(g_out, "\n\nCalculation complete\nClover is finishing\nWall clock %f\nFirst step overhead %f\n", wall_clock, first_step - second_step);
            printf("Wall clock %f\nFirst step overhead %f\n", wall_clock, first_step - second_step);

            if (profiler_on) {
                // TODO
            }

            // clover_finalize();
            break;
        }
    }
}
