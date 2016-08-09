
#include "../definitions_c.h"

#if defined(USE_KOKKOS)
#include "kokkos/accelerate.c"

void accelerate_adaptor()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        accelerate_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1,
            dt);
        g.compute();
    }
}

#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/accelerate_kernel.c"

void accelerate_adaptor()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax + 1,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax + 1,
            {
                accelerate_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin, chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin, chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.xarea,
                    chunk.tiles[tile].field.yarea,
                    chunk.tiles[tile].field.volume,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.viscosity,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.yvel1,
                    dt);
            });
        }
    }
}

#endif

#if defined(USE_OPENCL)
#include "../definitions_c.h"
void accelerate_adaptor()
{
    cl::Kernel accelerate_kernel(openclProgram, "accelerate_kernel");
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;
        accelerate_kernel.setArg(0,  xmin);
        accelerate_kernel.setArg(1,  xmax);
        accelerate_kernel.setArg(2,  ymin);
        accelerate_kernel.setArg(3,  ymax);
        accelerate_kernel.setArg(4,  *chunk.tiles[tile].field.d_xarea);
        accelerate_kernel.setArg(5,  *chunk.tiles[tile].field.d_yarea);
        accelerate_kernel.setArg(6,  *chunk.tiles[tile].field.d_volume);
        accelerate_kernel.setArg(7,  *chunk.tiles[tile].field.d_density0);
        accelerate_kernel.setArg(8,  *chunk.tiles[tile].field.d_pressure);
        accelerate_kernel.setArg(9,  *chunk.tiles[tile].field.d_viscosity);
        accelerate_kernel.setArg(10, *chunk.tiles[tile].field.d_xvel0);
        accelerate_kernel.setArg(11, *chunk.tiles[tile].field.d_yvel0);
        accelerate_kernel.setArg(12, *chunk.tiles[tile].field.d_xvel1);
        accelerate_kernel.setArg(13, *chunk.tiles[tile].field.d_yvel1);
        accelerate_kernel.setArg(14, dt);
        openclQueue.enqueueNDRangeKernel(
            accelerate_kernel,
            cl::NullRange,
            cl::NDRange(xmax - xmin + 1, ymax - ymin + 1),
            cl::NullRange);
    }
    if (profiler_on)
        openclQueue.finish();
}

#endif