#include "../definitions_c.h"
#include "../kernels/calc_dt_kernel_c.c"

#if defined(USE_KOKKOS)

void calc_dt_adaptor(int tile, double* local_dt)
{
    // TODO kokkos reduction
    double dt = g_big;
    for (int k = chunk.tiles[tile].t_ymin; k <= chunk.tiles[tile].t_ymax; k++) {
        for (int j = chunk.tiles[tile].t_xmin; j <= chunk.tiles[tile].t_xmax; j++) {
            double val = calc_dt_kernel_c_(
                             j, k,
                             chunk.tiles[tile].t_xmin,
                             chunk.tiles[tile].t_xmax,
                             chunk.tiles[tile].t_ymin,
                             chunk.tiles[tile].t_ymax,
                             chunk.tiles[tile].field.xarea,
                             chunk.tiles[tile].field.yarea,
                             chunk.tiles[tile].field.celldx,
                             chunk.tiles[tile].field.celldy,
                             chunk.tiles[tile].field.volume,
                             chunk.tiles[tile].field.density0,
                             chunk.tiles[tile].field.energy0,
                             chunk.tiles[tile].field.pressure,
                             chunk.tiles[tile].field.viscosity,
                             chunk.tiles[tile].field.soundspeed,
                             chunk.tiles[tile].field.xvel0,
                             chunk.tiles[tile].field.yvel0,
                             chunk.tiles[tile].field.work_array1
                         );
            if (val < dt)
                dt = val;
        }
    }
    *local_dt = dt;
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

void calc_dt_adaptor(int tile, double* local_dt)
{
    double dt = g_big;
    #pragma omp parallel for reduction(min:dt)
    for (int k = chunk.tiles[tile].t_ymin; k <= chunk.tiles[tile].t_ymax; k++) {
        for (int j = chunk.tiles[tile].t_xmin; j <= chunk.tiles[tile].t_xmax; j++) {
            double val = calc_dt_kernel_c_(
                             j, k,
                             chunk.tiles[tile].t_xmin,
                             chunk.tiles[tile].t_xmax,
                             chunk.tiles[tile].t_ymin,
                             chunk.tiles[tile].t_ymax,
                             chunk.tiles[tile].field.xarea,
                             chunk.tiles[tile].field.yarea,
                             chunk.tiles[tile].field.celldx,
                             chunk.tiles[tile].field.celldy,
                             chunk.tiles[tile].field.volume,
                             chunk.tiles[tile].field.density0,
                             chunk.tiles[tile].field.energy0,
                             chunk.tiles[tile].field.pressure,
                             chunk.tiles[tile].field.viscosity,
                             chunk.tiles[tile].field.soundspeed,
                             chunk.tiles[tile].field.xvel0,
                             chunk.tiles[tile].field.yvel0,
                             chunk.tiles[tile].field.work_array1
                         );
            if (val < dt)
                dt = val;
        }
    }
    *local_dt = dt;
}
#endif

#if defined(USE_OPENCL)

void calc_dt_adaptor(int tile, double* local_dt)
{
    double dt = g_big;
    for (int k = chunk.tiles[tile].t_ymin; k <= chunk.tiles[tile].t_ymax; k++) {
        for (int j = chunk.tiles[tile].t_xmin; j <= chunk.tiles[tile].t_xmax; j++) {
            double val = calc_dt_kernel_c_(
                             j, k,
                             chunk.tiles[tile].t_xmin,
                             chunk.tiles[tile].t_xmax,
                             chunk.tiles[tile].t_ymin,
                             chunk.tiles[tile].t_ymax,
                             chunk.tiles[tile].field.xarea,
                             chunk.tiles[tile].field.yarea,
                             chunk.tiles[tile].field.celldx,
                             chunk.tiles[tile].field.celldy,
                             chunk.tiles[tile].field.volume,
                             chunk.tiles[tile].field.density0,
                             chunk.tiles[tile].field.energy0,
                             chunk.tiles[tile].field.pressure,
                             chunk.tiles[tile].field.viscosity,
                             chunk.tiles[tile].field.soundspeed,
                             chunk.tiles[tile].field.xvel0,
                             chunk.tiles[tile].field.yvel0,
                             chunk.tiles[tile].field.work_array1
                         );
            if (val < dt)
                dt = val;
        }
    }
    *local_dt = dt;
}
#endif
