
#if defined(USE_KOKKOS)

#include "kokkos/pdv.cpp"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    if (predict) {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            pdv_predict_functor f(
                chunk.tiles[tile],
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                dt);
            f.compute();
        }
    } else {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            pdv_no_predict_functor f(
                chunk.tiles[tile],
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                dt);
            f.compute();
        }
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/PdV_kernel_c.c"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    #pragma omp parallel
    {
        if (predict)
        {
            for (int tile = 0; tile < tiles_per_chunk; tile++) {
                DOUBLEFOR(
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax, {
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
                });
            }
        } else {
            for (int tile = 0; tile < tiles_per_chunk; tile++)
            {
                DOUBLEFOR(
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax, {
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
                });
            }
        }
    }
}
#endif

#if defined(USE_CUDA)

#include "../kernels/PdV_kernel_c.c"

__global__ void pdv_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const double* xarea,
    const double* yarea,
    const double* volume,
    const double* density0,
    double*       density1,
    const double* energy0,
    double*       energy1,
    const double* pressure,
    const double* viscosity,
    const double* xvel0,
    const double* xvel1,
    const double* yvel0,
    const double* yvel1,
    double*       volume_change,
    int predict)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;
    if (j <= x_max && k <= y_max)
        if (predict == 0) {
            pdv_kernel_predict_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                dt,
                xarea,
                yarea,
                volume,
                density0,
                density1,
                energy0,
                energy1,
                pressure,
                viscosity,
                xvel0,
                xvel1,
                yvel0,
                yvel1,
                volume_change);
        } else {
            pdv_kernel_no_predict_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                dt,
                xarea,
                yarea,
                volume,
                density0,
                density1,
                energy0,
                energy1,
                pressure,
                viscosity,
                xvel0,
                xvel1,
                yvel0,
                yvel1,
                volume_change);
        }
}

void pdv(struct chunk_type chunk, bool predict, double dt)
{

    for (int tile = 0; tile < tiles_per_chunk; tile++) {

        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        dim3 size = numBlocks(
                        dim3((x_max) - (x_min) + 1,
                             (y_max) - (y_min) + 1),
                        pdv_kernel_blocksize);
        pdv_kernel <<< size, pdv_kernel_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            dt,
            chunk.tiles[tile].field.d_xarea,
            chunk.tiles[tile].field.d_yarea,
            chunk.tiles[tile].field.d_volume,
            chunk.tiles[tile].field.d_density0,
            chunk.tiles[tile].field.d_density1,
            chunk.tiles[tile].field.d_energy0,
            chunk.tiles[tile].field.d_energy1,
            chunk.tiles[tile].field.d_pressure,
            chunk.tiles[tile].field.d_viscosity,
            chunk.tiles[tile].field.d_xvel0,
            chunk.tiles[tile].field.d_xvel1,
            chunk.tiles[tile].field.d_yvel0,
            chunk.tiles[tile].field.d_yvel1,
            chunk.tiles[tile].field.d_work_array1,
            predict ? 0 : 1);
    }

    if (profiler_on)
        cudaDeviceSynchronize();
}
#endif

#if defined(USE_OPENCL)
#include "../definitions_c.h"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    cl::Kernel pdv_kernel(openclProgram, "pdv_kernel");
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;
        pdv_kernel.setArg(0,  xmin);
        pdv_kernel.setArg(1,  xmax);
        pdv_kernel.setArg(2,  ymin);
        pdv_kernel.setArg(3,  ymax);
        pdv_kernel.setArg(4, dt);

        pdv_kernel.setArg(5, *chunk.tiles[tile].field.d_xarea);
        pdv_kernel.setArg(6, *chunk.tiles[tile].field.d_yarea);
        pdv_kernel.setArg(7, *chunk.tiles[tile].field.d_volume);
        pdv_kernel.setArg(8, *chunk.tiles[tile].field.d_density0);
        pdv_kernel.setArg(9, *chunk.tiles[tile].field.d_density1);
        pdv_kernel.setArg(10, *chunk.tiles[tile].field.d_energy0);
        pdv_kernel.setArg(11, *chunk.tiles[tile].field.d_energy1);
        pdv_kernel.setArg(12, *chunk.tiles[tile].field.d_pressure);
        pdv_kernel.setArg(13, *chunk.tiles[tile].field.d_viscosity);
        pdv_kernel.setArg(14, *chunk.tiles[tile].field.d_xvel0);
        pdv_kernel.setArg(15, *chunk.tiles[tile].field.d_xvel1);
        pdv_kernel.setArg(16, *chunk.tiles[tile].field.d_yvel0);
        pdv_kernel.setArg(17, *chunk.tiles[tile].field.d_yvel1);
        pdv_kernel.setArg(18, *chunk.tiles[tile].field.d_work_array1);
        if (predict) {
            pdv_kernel.setArg(19, 0);
        } else {
            pdv_kernel.setArg(19, 1);
        }
        openclQueue.enqueueNDRangeKernel(
            pdv_kernel,
            cl::NullRange,
            cl::NDRange(xmax - xmin + 1, ymax - ymin + 1),
            pdv_kernel_local_size);
    }

    if (profiler_on)
        openclQueue.finish();
}
#endif
