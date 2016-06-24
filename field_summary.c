#include "field_summary.h"
#include "definitions_c.h"
#include "field_summary_kernel_c.c"
#include "timer_c.h"
#include "stdlib.h"
#include "ideal_gas.h"

void field_summary()
{
    double kernel_time,
           vol, mass, ie, ke, press;
    fprintf(g_out, "\nTime %.13f\n", _time); // TODO
    fprintf(g_out, "%13s%16s%16s%16s%16s%16s%16s%16s\n", "", "Volume", "Mass", "Density", "Pressure", "Internal Energy", "Kinetic Energy", "Total Energy");

    if (profiler_on)
        kernel_time = timer();

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        ideal_gas(tile, false);
    }

    if (profiler_on) {
        profiler.ideal_gas += timer() - kernel_time;
        kernel_time = timer();
    }


    double t_vol = 0.0;
    double t_mass = 0.0;
    double t_ie = 0.0;
    double t_ke = 0.0;
    double t_press = 0.0;

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        field_summary_kernel_c_(
            &chunk.tiles[tile].t_xmin,
            &chunk.tiles[tile].t_xmax,
            &chunk.tiles[tile].t_ymin,
            &chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.volume,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.energy0,
            chunk.tiles[tile].field.pressure,
            chunk.tiles[tile].field.xvel0,
            chunk.tiles[tile].field.yvel0,
            &vol, &mass, &ie, &ke, &press);
        t_vol = t_vol + vol;
        t_mass = t_mass + mass;
        t_ie = t_ie + ie;
        t_ke = t_ke + ke;
        t_press = t_press + press;
    }
    if (profiler_on) profiler.summary += timer() - kernel_time;

    fprintf(g_out, "%6s%7d%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e%16.7e\n\n", " step:", step, vol, mass, mass / vol, press / vol, ie, ke, ie + ke);
    double qa_diff;
    if (complete) {
        if (test_problem >= 1) {
            if (test_problem == 1) qa_diff = fabs((100.0 * (ke / 1.82280367310258)) - 100.0);
            if (test_problem == 2) qa_diff = fabs((100.0 * (ke / 1.19316898756307)) - 100.0);
            if (test_problem == 3) qa_diff = fabs((100.0 * (ke / 2.58984003503994)) - 100.0);
            if (test_problem == 4) qa_diff = fabs((100.0 * (ke / 0.307475452287895)) - 100.0);
            if (test_problem == 5) qa_diff = fabs((100.0 * (ke / 4.85350315783719)) - 100.0);

            fprintf(stdout, "%s %d %s %16.7e %s\n", "Test problem", test_problem, "is within", qa_diff, "% of the expected solution");
            fprintf(g_out, "%s %d %s %16.7e %s\n", "Test problem", test_problem, "is within", qa_diff, "% of the expected solution");
            if (qa_diff < 0.001) {
                fprintf(stdout, "This test is considered PASSED\n");
                fprintf(g_out, "This test is considered PASSED\n");
            } else {
                fprintf(stdout, "This test is considered NOT PASSED\n");
                fprintf(g_out, "This test is considered NOT PASSED\n");
            }
        }
    }
}


void clover_sum(double val)
{
    // TODO MPI stuff
}
