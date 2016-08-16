
#if defined(USE_KOKKOS)
#include "../definitions_c.h"
#include "kokkos/flux_calc.cpp"
void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        flux_calc_x_functor f1(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            dt);
        f1.compute();

        flux_calc_y_functor f2(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1,
            dt);
        f2.compute();
        fence();
    }

}
#endif


#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../definitions_c.h"
#include "../kernels/flux_calc_kernel_c.c"

void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1, {
                flux_calc_x_kernel(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    dt,
                    chunk.tiles[tile].field.xarea,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.vol_flux_x);
            });

            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax + 1,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                flux_calc_y_kernel(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    dt,
                    chunk.tiles[tile].field.yarea,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.yvel1,
                    chunk.tiles[tile].field.vol_flux_y);
            });
        }
    }

}
#endif

#if defined(USE_CUDA)
#include "../definitions_c.h"
#include "../kernels/flux_calc_kernel_c.c"

__global__ void flux_calc_x_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    field_2d_t vol_flux_x)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max)
        flux_calc_x_kernel(
            j, k,
            x_min, x_max,
            y_min, y_max,
            dt,
            xarea,
            xvel0,
            xvel1,
            vol_flux_x);
}
__global__ void flux_calc_y_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    field_2d_t vol_flux_x)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max)
        flux_calc_y_kernel(
            j, k,
            x_min, x_max,
            y_min, y_max,
            dt,
            xarea,
            xvel0,
            xvel1,
            vol_flux_x);
}

void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {

        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        dim3 size1 = numBlocks(
                         dim3((x_max + 1) - (x_min) + 1,
                              (y_max) - (y_min) + 1),
                         flux_calc_x_blocksize);
        flux_calc_x_kernel <<< size1, flux_calc_x_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            dt,
            chunk.tiles[tile].field.d_xarea,
            chunk.tiles[tile].field.d_xvel0,
            chunk.tiles[tile].field.d_xvel1,
            chunk.tiles[tile].field.d_vol_flux_x);

        dim3 size2 = numBlocks(
                         dim3((x_max) - (x_min) + 1,
                              (y_max + 1) - (y_min) + 1),
                         flux_calc_y_blocksize);
        flux_calc_y_kernel <<< size2, flux_calc_y_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            dt,
            chunk.tiles[tile].field.d_yarea,
            chunk.tiles[tile].field.d_yvel0,
            chunk.tiles[tile].field.d_yvel1,
            chunk.tiles[tile].field.d_vol_flux_y);
    }

    if (profiler_on)
        cudaDeviceSynchronize();
}
#endif

#if defined(USE_OPENCL)
#include "../definitions_c.h"
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/flux_calc_kernel_c.c"

void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        cl::Kernel flux_calc_x(openclProgram, "flux_calc_x_kernel_");
        flux_calc_x.setArg(0,  x_min);
        flux_calc_x.setArg(1,  x_max);
        flux_calc_x.setArg(2,  y_min);
        flux_calc_x.setArg(3,  y_max);

        flux_calc_x.setArg(4, dt);
        flux_calc_x.setArg(5, *chunk.tiles[tile].field.d_xarea);
        flux_calc_x.setArg(6, *chunk.tiles[tile].field.d_xvel0);
        flux_calc_x.setArg(7, *chunk.tiles[tile].field.d_xvel1);
        flux_calc_x.setArg(8, *chunk.tiles[tile].field.d_vol_flux_x);
        openclQueue.enqueueNDRangeKernel(
            flux_calc_x,
            cl::NullRange,
            cl::NDRange((x_max + 1) - (x_min) + 1, (y_max) - (y_min) + 1),
            flux_calc_x_local_size);

        cl::Kernel flux_calc_y(openclProgram, "flux_calc_y_kernel_");
        flux_calc_y.setArg(0,  x_min);
        flux_calc_y.setArg(1,  x_max);
        flux_calc_y.setArg(2,  y_min);
        flux_calc_y.setArg(3,  y_max);

        flux_calc_y.setArg(4, dt);
        flux_calc_y.setArg(5, *chunk.tiles[tile].field.d_yarea);
        flux_calc_y.setArg(6, *chunk.tiles[tile].field.d_yvel0);
        flux_calc_y.setArg(7, *chunk.tiles[tile].field.d_yvel1);
        flux_calc_y.setArg(8, *chunk.tiles[tile].field.d_vol_flux_y);
        openclQueue.enqueueNDRangeKernel(
            flux_calc_y,
            cl::NullRange,
            cl::NDRange((x_max) - (x_min) + 1, (y_max + 1) - (y_min) + 1),
            flux_calc_y_local_size);
    }
    openclQueue.finish();
}
#endif
