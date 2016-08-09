
#if defined(USE_KOKKOS)
#include "../definitions_c.h"
#include "../kernels/ftocmacros.h"
#include "kokkos/viscosity.cpp"
void viscosity(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        viscosity_functor f(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax);
        f.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../definitions_c.h"
#include "../kernels/ftocmacros.h"
#include <math.h>
#include "../kernels/viscosity_kernel_c.c"

void viscosity(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                viscosity_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.celldx,
                    chunk.tiles[tile].field.celldy,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.viscosity,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.yvel0);
            });
        }
    }
}
#endif

#if defined(USE_OPENCL)
#include "../definitions_c.h"
#include "../kernels/ftocmacros.h"
#include <math.h>
#include "../kernels/viscosity_kernel_c.c"

void viscosity(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        cl::Kernel viscosity(openclProgram, "viscosity_kernel");
        viscosity.setArg(0,  x_min);
        viscosity.setArg(1,  x_max);
        viscosity.setArg(2,  y_min);
        viscosity.setArg(3,  y_max);

        viscosity.setArg(4, *chunk.tiles[tile].field.d_celldx);
        viscosity.setArg(5, *chunk.tiles[tile].field.d_celldy);
        viscosity.setArg(6, *chunk.tiles[tile].field.d_density0);
        viscosity.setArg(7, *chunk.tiles[tile].field.d_pressure);
        viscosity.setArg(8, *chunk.tiles[tile].field.d_viscosity);
        viscosity.setArg(9, *chunk.tiles[tile].field.d_xvel0);
        viscosity.setArg(10, *chunk.tiles[tile].field.d_yvel0);
        openclQueue.enqueueNDRangeKernel(
            viscosity,
            cl::NullRange,
            cl::NDRange((x_max) - (x_min) + 1, (y_max) - (y_min) + 1),
            viscosity_local_size);
    }
    if (profiler_on)
        openclQueue.finish();
}
#endif