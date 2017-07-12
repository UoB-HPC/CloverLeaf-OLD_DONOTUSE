
#if defined(USE_KOKKOS)
#include "../definitions_c.h"
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
            chunk.tiles[tile].field.d_density1,
            chunk.tiles[tile].field.d_energy1);
        g.compute();
    } else {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.d_density0,
            chunk.tiles[tile].field.d_energy0);
        g.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include <math.h>
#include "../kernels/ideal_gas_kernel_c.cc"

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

#if defined(USE_CUDA)
#include <math.h>
#include "../kernels/ideal_gas_kernel_c.c"

__global__ void ideal_gas_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t   density,
    const_field_2d_t   energy,
    field_2d_t         pressure,
    field_2d_t         soundspeed)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max)
        ideal_gas_kernel_c_(
            j, k,
            x_min, x_max,
            y_min, y_max,
            density,
            energy,
            pressure,
            soundspeed);
}

void ideal_gas_adaptor(int tile, bool predict)
{
    int x_min = chunk.tiles[tile].t_xmin,
        x_max = chunk.tiles[tile].t_xmax,
        y_min = chunk.tiles[tile].t_ymin,
        y_max = chunk.tiles[tile].t_ymax;

    dim3 size = numBlocks(
                    dim3((x_max) - (x_min) + 1,
                         (y_max) - (y_min) + 1),
                    ideal_gas_blocksize);
    if (predict) {
        ideal_gas_kernel <<< size, ideal_gas_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            chunk.tiles[tile].field.d_density1,
            chunk.tiles[tile].field.d_energy1,
            chunk.tiles[tile].field.d_pressure,
            chunk.tiles[tile].field.d_soundspeed);
    } else {
        ideal_gas_kernel <<< size, ideal_gas_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            chunk.tiles[tile].field.d_density0,
            chunk.tiles[tile].field.d_energy0,
            chunk.tiles[tile].field.d_pressure,
            chunk.tiles[tile].field.d_soundspeed);
    }

    if (profiler_on)
        cudaDeviceSynchronize();
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
            calcGlobalSize(cl::NDRange(xmax - xmin + 1, ymax - ymin + 1),
                           ideal_gas_local_size),
            ideal_gas_local_size);
    }

    if (profiler_on)
        openclQueue.finish();
}
#endif
