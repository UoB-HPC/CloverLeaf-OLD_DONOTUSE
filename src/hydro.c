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
#include "accelerate.h"
#include "clover.h"
#include <stdlib.h>

// #include <xmmintrin.h>

int maxloc(double* arr, int size)
{
    double maxval = arr[0];
    int maxloc = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > maxval) {
            maxval = arr[i];
            maxloc = i;
        }
    }
    return maxloc;
}

void hydro()
{
    double timerstart = timer(),
           step_time,
           first_step = 0.0, second_step = 0.0,
           wall_clock;
    double* totals = (double*)calloc(parallel.max_task, sizeof(double));


    // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
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

            BOSSPRINT(g_out, "\n\nCalculation complete\nClover is finishing\nWall clock %f\nFirst step overhead %f\n", wall_clock, first_step - second_step);
            BOSSPRINT(stdout, "Wall clock %f\nFirst step overhead %f\n", wall_clock, first_step - second_step);

            if (profiler_on) {

                //     ! First we need to find the maximum kernel time for each task. This
                //     ! seems to work better than finding the maximum time for each kernel and
                //     ! adding it up, which always gives over 100%. I think this is because it
                //     ! does not take into account compute overlaps before syncronisations
                //     ! caused by halo exhanges.
                //     kernel_total=profiler%timestep+profiler%ideal_gas+profiler%viscosity+profiler%PdV          &
                //         +profiler%revert+profiler%acceleration+profiler%flux+profiler%cell_advection   &
                //         +profiler%mom_advection+profiler%reset+profiler%summary+profiler%visit         &
                //         +profiler%tile_halo_exchange+profiler%self_halo_exchange+profiler%mpi_halo_exchange
                double kernel_total = profiler.timestep + profiler.ideal_gas + profiler.viscosity +
                                      profiler.PdV + profiler.revert + profiler.acceleration + profiler.flux + profiler.cell_advection +
                                      profiler.mom_advection + profiler.reset + profiler.summary + profiler.visit +
                                      profiler.tile_halo_exchange + profiler.self_halo_exchange + profiler.mpi_halo_exchange;
                //     CALL clover_allgather(kernel_total,totals)
                clover_allgather(&kernel_total, totals);
                //     ! So then what I do is use the individual kernel times for the
                //     ! maximum kernel time task for the profile print
                //     loc=MAXLOC(totals)
                int loc = maxloc(totals, parallel.max_task);
                //     kernel_total=totals(loc(1))
                kernel_total = totals[loc];
                //     CALL clover_allgather(profiler%timestep,totals)
                clover_allgather(&profiler.timestep, totals);
                //     profiler%timestep=totals(loc(1))
                profiler.timestep = totals[loc];
                //     CALL clover_allgather(profiler%ideal_gas,totals)
                clover_allgather(&profiler.ideal_gas, totals);
                //     profiler%ideal_gas=totals(loc(1))
                profiler.ideal_gas = totals[loc];
                //     CALL clover_allgather(profiler%viscosity,totals)
                clover_allgather(&profiler.viscosity, totals);
                //     profiler%viscosity=totals(loc(1))
                profiler.viscosity = totals[loc];
                //     CALL clover_allgather(profiler%PdV,totals)
                clover_allgather(&profiler.PdV, totals);
                //     profiler%PdV=totals(loc(1))
                profiler.PdV = totals[loc];
                //     CALL clover_allgather(profiler%revert,totals)
                clover_allgather(&profiler.revert, totals);
                //     profiler%revert=totals(loc(1))
                profiler.revert = totals[loc];
                //     CALL clover_allgather(profiler%acceleration,totals)
                clover_allgather(&profiler.acceleration, totals);
                //     profiler%acceleration=totals(loc(1))
                profiler.acceleration = totals[loc];
                //     CALL clover_allgather(profiler%flux,totals)
                clover_allgather(&profiler.flux, totals);
                //     profiler%flux=totals(loc(1))
                profiler.flux = totals[loc];
                //     CALL clover_allgather(profiler%cell_advection,totals)
                clover_allgather(&profiler.cell_advection, totals);
                //     profiler%cell_advection=totals(loc(1))
                profiler.cell_advection = totals[loc];
                //     CALL clover_allgather(profiler%mom_advection,totals)
                clover_allgather(&profiler.mom_advection, totals);
                //     profiler%mom_advection=totals(loc(1))
                profiler.mom_advection = totals[loc];
                //     CALL clover_allgather(profiler%reset,totals)
                clover_allgather(&profiler.reset, totals);
                //     profiler%reset=totals(loc(1))
                profiler.reset = totals[loc];
                //     CALL clover_allgather(profiler%tile_halo_exchange,totals)
                clover_allgather(&profiler.tile_halo_exchange, totals);
                //     profiler%tile_halo_exchange=totals(loc(1))
                profiler.tile_halo_exchange = totals[loc];
                //     CALL clover_allgather(profiler%self_halo_exchange,totals)
                clover_allgather(&profiler.self_halo_exchange, totals);
                //     profiler%self_halo_exchange=totals(loc(1))
                profiler.self_halo_exchange = totals[loc];
                //     CALL clover_allgather(profiler%mpi_halo_exchange,totals)
                clover_allgather(&profiler.mpi_halo_exchange, totals);
                //     profiler%mpi_halo_exchange=totals(loc(1))
                profiler.mpi_halo_exchange = totals[loc];
                //     CALL clover_allgather(profiler%summary,totals)
                clover_allgather(&profiler.summary, totals);
                //     profiler%summary=totals(loc(1))
                profiler.summary = totals[loc];
                //     CALL clover_allgather(profiler%visit,totals)
                clover_allgather(&profiler.visit, totals);
                //     profiler%visit=totals(loc(1))
                profiler.visit = totals[loc];

                //     IF ( parallel%boss ) THEN
                BOSSPRINT(stdout, "Profiler Output                 Time            Percentage\n");
                //         WRITE(g_out,*)
                //         WRITE(g_out,'(a58,2f16.4)')"Profiler Output                 Time            Percentage"
                //         WRITE(g_out,'(a23,2f16.4)')"Timestep              :",profiler%timestep,&
                //             100.0*(profiler%timestep/wall_clock)
                BOSSPRINT(stdout, "Timestep              :%16.4f %16.4f\n", profiler.timestep, 100.0 * (profiler.timestep / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Ideal Gas             :",profiler%ideal_gas,&
                //             100.0*(profiler%ideal_gas/wall_clock)
                BOSSPRINT(stdout, "ideal_gas             :%16.4f %16.4f\n", profiler.ideal_gas, 100.0 * (profiler.ideal_gas / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Viscosity             :",profiler%viscosity,&
                //             100.0*(profiler%viscosity/wall_clock)
                BOSSPRINT(stdout, "viscosity             :%16.4f %16.4f\n", profiler.viscosity, 100.0 * (profiler.viscosity / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"PdV                   :",profiler%PdV,&
                //             100.0*(profiler%PdV/wall_clock)
                BOSSPRINT(stdout, "PdV                   :%16.4f %16.4f\n", profiler.PdV, 100.0 * (profiler.PdV / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Revert                :",profiler%revert,&
                //             100.0*(profiler%revert/wall_clock)
                BOSSPRINT(stdout, "revert                :%16.4f %16.4f\n", profiler.revert, 100.0 * (profiler.revert / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Acceleration          :",profiler%acceleration,&
                //             100.0*(profiler%acceleration/wall_clock)
                BOSSPRINT(stdout, "acceleration          :%16.4f %16.4f\n", profiler.acceleration, 100.0 * (profiler.acceleration / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Fluxes                :",profiler%flux,&
                //             100.0*(profiler%flux/wall_clock)
                BOSSPRINT(stdout, "flux                  :%16.4f %16.4f\n", profiler.flux, 100.0 * (profiler.flux / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Cell Advection        :",profiler%cell_advection,&
                //             100.0*(profiler%cell_advection/wall_clock)
                BOSSPRINT(stdout, "cell_advection        :%16.4f %16.4f\n", profiler.cell_advection, 100.0 * (profiler.cell_advection / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Momentum Advection    :",profiler%mom_advection,&
                //             100.0*(profiler%mom_advection/wall_clock)
                BOSSPRINT(stdout, "mom_advection         :%16.4f %16.4f\n", profiler.mom_advection, 100.0 * (profiler.mom_advection / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Reset                 :",profiler%reset,&
                //             100.0*(profiler%reset/wall_clock)
                BOSSPRINT(stdout, "reset                 :%16.4f %16.4f\n", profiler.reset, 100.0 * (profiler.reset / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Summary               :",profiler%summary,&
                //             100.0*(profiler%summary/wall_clock)
                BOSSPRINT(stdout, "summary               :%16.4f %16.4f\n", profiler.summary, 100.0 * (profiler.summary / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Visit                 :",profiler%visit,&
                //             100.0*(profiler%visit/wall_clock)
                BOSSPRINT(stdout, "visit                 :%16.4f %16.4f\n", profiler.visit, 100.0 * (profiler.visit / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Tile Halo Exchange    :",profiler%tile_halo_exchange,&
                //             100.0*(profiler%tile_halo_exchange/wall_clock)
                BOSSPRINT(stdout, "tile_halo_exchange    :%16.4f %16.4f\n", profiler.tile_halo_exchange, 100.0 * (profiler.tile_halo_exchange / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Self Halo Exchange    :",profiler%self_halo_exchange,&
                //             100.0*(profiler%self_halo_exchange/wall_clock)
                BOSSPRINT(stdout, "self_halo_exchange    :%16.4f %16.4f\n", profiler.self_halo_exchange, 100.0 * (profiler.self_halo_exchange / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"MPI Halo Exchange     :",profiler%mpi_halo_exchange,&
                //             100.0*(profiler%mpi_halo_exchange/wall_clock)
                BOSSPRINT(stdout, "mpi_halo_exchange     :%16.4f %16.4f\n", profiler.mpi_halo_exchange, 100.0 * (profiler.mpi_halo_exchange / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"Total                 :",kernel_total,&
                //             100.0*(kernel_total/wall_clock)
                BOSSPRINT(stdout, "kernel_total          :%16.4f %16.4f\n", kernel_total, 100.0 * (kernel_total / wall_clock));
                //         WRITE(g_out,'(a23,2f16.4)')"The Rest              :",wall_clock-kernel_total,&
                //             100.0*(wall_clock-kernel_total)/wall_clock
                BOSSPRINT(stdout, "the rest              :%16.4f %16.4f\n", wall_clock - kernel_total, 100.0 * ((wall_clock - kernel_total) / wall_clock));
                //     ENDIF
            }

            // clover_finalize();
            break;
        }
    }
}

// for debugging state
// for (int tile = 0; tile < tiles_per_chunk; tile++) {
//     debug_kernel_(
//         &tile,
//         &chunk.tiles[tile].t_xmin,
//         &chunk.tiles[tile].t_xmax,
//         &chunk.tiles[tile].t_ymin,
//         &chunk.tiles[tile].t_ymax,
//         chunk.tiles[tile].field.density0,
//         chunk.tiles[tile].field.density1,
//         chunk.tiles[tile].field.energy0,
//         chunk.tiles[tile].field.energy1,
//         chunk.tiles[tile].field.pressure,
//         chunk.tiles[tile].field.viscosity,
//         chunk.tiles[tile].field.soundspeed,
//         chunk.tiles[tile].field.xvel0,
//         chunk.tiles[tile].field.xvel1,
//         chunk.tiles[tile].field.yvel0,
//         chunk.tiles[tile].field.yvel1,
//         chunk.tiles[tile].field.vol_flux_x,
//         chunk.tiles[tile].field.vol_flux_y,
//         chunk.tiles[tile].field.mass_flux_x,
//         chunk.tiles[tile].field.mass_flux_y,
//         chunk.tiles[tile].field.xarea,
//         chunk.tiles[tile].field.yarea,
//         chunk.tiles[tile].field.volume
//     );
// };
