
#if defined(USE_KOKKOS)
#include "kokkos/ideal_gas.cpp"

void ideal_gas_adaptor(int tile, bool predict)
{
    if (predict) {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density1,
            chunk.tiles[tile].field.energy1);
        g.compute();
    } else {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.energy0);
        g.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include <math.h>
#include "../kernels/ideal_gas_kernel_c.c"

void ideal_gas_adaptor(int tile, bool predict)
{
    #pragma omp parallel
    {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
            if (predict)
            {
                ideal_gas_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy1,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.soundspeed
                );
            } else {
                ideal_gas_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.soundspeed
                );
            }
        });
    }
}
#endif

#if defined(USE_OPENCL)
#include "../definitions_c.h"
void ideal_gas_adaptor(int tile, bool predict)
{
    cl::Kernel ideal_gas(openclProgram, "ideal_gas_kernel");
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;

        ideal_gas.setArg(0,  xmin);
        ideal_gas.setArg(1,  xmax);
        ideal_gas.setArg(2,  ymin);
        ideal_gas.setArg(3,  ymax);
        if (predict) {
            ideal_gas.setArg(4, *chunk.tiles[tile].field.d_density1);
            ideal_gas.setArg(5, *chunk.tiles[tile].field.d_energy1);
        } else {
            ideal_gas.setArg(4, *chunk.tiles[tile].field.d_density0);
            ideal_gas.setArg(5, *chunk.tiles[tile].field.d_energy0);
        }
        ideal_gas.setArg(6, *chunk.tiles[tile].field.d_pressure);
        ideal_gas.setArg(7, *chunk.tiles[tile].field.d_soundspeed);
        openclQueue.enqueueNDRangeKernel(
            ideal_gas,
            cl::NullRange,
            cl::NDRange(xmax - xmin + 1, ymax - ymin + 1),
            cl::NullRange);
    }

    if (profiler_on)
        openclQueue.finish();
}
#endif