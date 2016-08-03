#include "../definitions_c.h"

#if defined(USE_KOKKOS)
#include "kokkos/calc_dt.cpp"
// #include "../kernels/calc_dt_kernel_c.c"
void calc_dt_adaptor(int tile, double* local_dt)
{
    // TODO kokkos reduction
    double dt = g_big;
    // for (int k = chunk.tiles[tile].t_ymin; k <= chunk.tiles[tile].t_ymax; k++) {
    calc_dt_functor f(chunk.tiles[tile],
                      chunk.tiles[tile].t_xmin,
                      chunk.tiles[tile].t_xmax,
                      chunk.tiles[tile].t_ymin,
                      chunk.tiles[tile].t_ymax);
    // double res;
    f.compute(dt);
    //     if (res < dt)
    //         dt = res;
    // }
    *local_dt = dt;
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/calc_dt_kernel_c.c"

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
    double min = g_big;
    cl::Kernel calc_dt(openclProgram, "calc_dt_kernel");
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        calc_dt.setArg(0,  x_min);
        calc_dt.setArg(1,  x_max);
        calc_dt.setArg(2,  y_min);
        calc_dt.setArg(3,  y_max);

        calc_dt.setArg(4, *chunk.tiles[tile].field.d_xarea);
        calc_dt.setArg(5, *chunk.tiles[tile].field.d_yarea);
        calc_dt.setArg(6, *chunk.tiles[tile].field.d_celldx);
        calc_dt.setArg(7, *chunk.tiles[tile].field.d_celldy);
        calc_dt.setArg(8, *chunk.tiles[tile].field.d_volume);
        calc_dt.setArg(9, *chunk.tiles[tile].field.d_density0);
        calc_dt.setArg(10, *chunk.tiles[tile].field.d_energy0);
        calc_dt.setArg(11, *chunk.tiles[tile].field.d_pressure);
        calc_dt.setArg(12, *chunk.tiles[tile].field.d_viscosity);
        calc_dt.setArg(13, *chunk.tiles[tile].field.d_soundspeed);
        calc_dt.setArg(14, *chunk.tiles[tile].field.d_xvel0);
        calc_dt.setArg(15, *chunk.tiles[tile].field.d_yvel0);
        calc_dt.setArg(16, *chunk.tiles[tile].field.d_work_array1);

        openclQueue.enqueueNDRangeKernel(calc_dt, cl::NullRange, cl::NDRange(x_max - x_min + 1, y_max - y_min + 1), cl::NullRange);
        openclQueue.finish();

        for (int k = y_min; k <= y_max; k++) {
            for (int j = x_min; j <= x_max; j++) {
                double val = WORK_ARRAY(chunk.tiles[tile].field.work_array1, j, k);
                if (val < min)
                    min = val;
            }
        }
    }

    *local_dt = min;
}
#endif
