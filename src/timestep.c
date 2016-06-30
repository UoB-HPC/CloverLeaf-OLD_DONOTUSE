#include "timestep.h"
#include "definitions_c.h"
#include "ideal_gas.h"
#include "viscosity.h"
#include "calc_dt.h"
#include "ftocmacros.h"
#include "report.h"
#include "timer_c.h"
#include "update_halo.h"
#include "string.h"
#include "clover.h"


void timestep()
{
    dt = g_big;
    int small = 0;
    double kernel_time = 0.0;

    int jldt, kldt;
    double dtlp;

    char dt_control[8], dtl_control[8];
    double xl_pos, yl_pos,
           x_pos = 0.0, y_pos = 0.0;

    if (profiler_on) kernel_time = timer();
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        ideal_gas(tile, false);
    }

    if (profiler_on) {
        profiler.ideal_gas += timer() - kernel_time;
    }

    int fields[NUM_FIELDS];
    for (int i = 0; i < NUM_FIELDS; i++) {
        fields[i] = 0;
    }

    fields[FIELD_PRESSURE] =
        fields[FIELD_ENERGY0] =
            fields[FIELD_DENSITY0] =
                fields[FIELD_XVEL0] =
                    fields[FIELD_YVEL0] = 1;

    update_halo(fields, 1);

    if (profiler_on) kernel_time = timer();
    viscosity();
    if (profiler_on)
        profiler.viscosity += timer() - kernel_time;


    for (int i = 0; i < NUM_FIELDS; i++) {
        fields[i] = 0;
    }
    fields[FIELD_VISCOSITY] = 1;
    update_halo(fields, 1);

    if (profiler_on) kernel_time = timer();

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        calc_dt(&tile, &dtlp, dtl_control, &xl_pos, &yl_pos, &jldt, &kldt);
        if (dtlp < dt) {
            dt = dtlp;
            // dt_control = dtl_control;
            memcpy(dt_control, dtl_control, 8 * sizeof(char));
            x_pos = xl_pos;
            y_pos = yl_pos;
            jdt = jldt;
            kdt = kldt;
        }
    }

    dt = MIN(dt, MIN((dtold * dtrise), dtmax));

    clover_min(&dt);
    if (profiler_on) profiler.timestep += timer() - kernel_time;

    if (dt < dtmin) small = 1;

    BOSSPRINT(g_out, " Step %d time %.7e control %-11s timestep %.5e %d, %d x=%.2e y=%.2e\n", step, _time, dt_control, dt, jdt, kdt, x_pos, y_pos);
    BOSSPRINT(stdout, " Step %d time %.7e control %-11s timestep %.5e %d, %d x=%.2e y=%.2e\n", step, _time, dt_control, dt, jdt, kdt, x_pos, y_pos);

    if (small == 1) {
        report_error("timestep", "small timestep");
    }

    dtold = dt;
}
