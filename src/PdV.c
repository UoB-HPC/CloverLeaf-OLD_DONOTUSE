#include "PdV.h"
#include "definitions_c.h"
#include "timer_c.h"
#include "kernels/PdV_kernel_c.c"
#include "ideal_gas.h"
#include "revert.h"
#include "update_halo.h"

void PdV(bool predict)
{
    // error_condition = 0;
    double kernel_time = 0.0;

    if (profiler_on) kernel_time = timer();

    #pragma omp parallel
    {
        if (predict) {
            for (int tile = 0; tile < tiles_per_chunk; tile++) {
                DOUBLEFOR(
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                ({
                    pdv_kernel_predict_c_(
                        j, k,
                        chunk.tiles[tile].t_xmin,
                        chunk.tiles[tile].t_xmax,
                        chunk.tiles[tile].t_ymin,
                        chunk.tiles[tile].t_ymax,
                        dt,
                        chunk.tiles[tile].field.xarea,
                        chunk.tiles[tile].field.yarea,
                        chunk.tiles[tile].field.volume,
                        chunk.tiles[tile].field.density0,
                        chunk.tiles[tile].field.density1,
                        chunk.tiles[tile].field.energy0,
                        chunk.tiles[tile].field.energy1,
                        chunk.tiles[tile].field.pressure,
                        chunk.tiles[tile].field.viscosity,
                        chunk.tiles[tile].field.xvel0,
                        chunk.tiles[tile].field.xvel1,
                        chunk.tiles[tile].field.yvel0,
                        chunk.tiles[tile].field.yvel1,
                        chunk.tiles[tile].field.work_array1);
                }));
            }
        } else {
            for (int tile = 0; tile < tiles_per_chunk; tile++) {
                DOUBLEFOR(
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                ({
                    pdv_kernel_no_predict_c_(
                        j, k,
                        chunk.tiles[tile].t_xmin,
                        chunk.tiles[tile].t_xmax,
                        chunk.tiles[tile].t_ymin,
                        chunk.tiles[tile].t_ymax,
                        dt,
                        chunk.tiles[tile].field.xarea,
                        chunk.tiles[tile].field.yarea,
                        chunk.tiles[tile].field.volume,
                        chunk.tiles[tile].field.density0,
                        chunk.tiles[tile].field.density1,
                        chunk.tiles[tile].field.energy0,
                        chunk.tiles[tile].field.energy1,
                        chunk.tiles[tile].field.pressure,
                        chunk.tiles[tile].field.viscosity,
                        chunk.tiles[tile].field.xvel0,
                        chunk.tiles[tile].field.xvel1,
                        chunk.tiles[tile].field.yvel0,
                        chunk.tiles[tile].field.yvel1,
                        chunk.tiles[tile].field.work_array1);
                }));
            }
        }
    }


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
