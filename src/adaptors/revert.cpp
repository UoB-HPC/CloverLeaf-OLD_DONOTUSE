#if defined(USE_KOKKOS)

#include "kokkos/revert.cpp"

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        struct tile_type t = chunk.tiles[tile];
        revert_functor f(t,
                         t.t_xmin,
                         t.t_xmax,
                         t.t_ymin,
                         t.t_ymax);
        f.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/revert_kernel_c.c"

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {

        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                revert_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.energy1);
            });
        }
    }

}
#endif

#if defined(USE_CUDA)

#include "../definitions_c.h"
#include "../kernels/revert_kernel_c.c"

__global__ void revert_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max)
        revert_kernel_c_(
            j, k,
            x_min,
            x_max,
            y_min,
            y_max,
            density0,
            density1,
            energy0,
            energy1);
}

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        dim3 size = numBlocks(
                        dim3((x_max) - (x_min) + 1,
                             (y_max) - (y_min) + 1),
                        revert_blocksize);
        revert_kernel <<< size, revert_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            chunk.tiles[tile].field.d_density0,
            chunk.tiles[tile].field.d_density1,
            chunk.tiles[tile].field.d_energy0,
            chunk.tiles[tile].field.d_energy1);
    }

    if (profiler_on)
        cudaDeviceSynchronize();
}
#endif


#if defined(USE_OPENCL)
#include "../kernels/revert_kernel_c.c"
#include "../definitions_c.h"

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        cl::Kernel revert(openclProgram, "revert_kernel");
        revert.setArg(0,  x_min);
        revert.setArg(1,  x_max);
        revert.setArg(2,  y_min);
        revert.setArg(3,  y_max);
        revert.setArg(4, *chunk.tiles[tile].field.d_density0);
        revert.setArg(5, *chunk.tiles[tile].field.d_density1);
        revert.setArg(6, *chunk.tiles[tile].field.d_energy0);
        revert.setArg(7, *chunk.tiles[tile].field.d_energy1);
        openclQueue.enqueueNDRangeKernel(
            revert,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max) - (x_min) + 1, (y_max) - (y_min) + 1),
                           revert_local_size),
            revert_local_size);
    }
    if (profiler_on)
        openclQueue.finish();
}
#endif