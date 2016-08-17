#include "../definitions_c.h"

#if defined(USE_KOKKOS)
#include "../kernels/ftocmacros.h"
#include "kokkos/calc_dt.cpp"
void calc_dt_adaptor(int tile, double* local_dt)
{
    double dt = g_big;
    calc_dt_functor f(chunk.tiles[tile],
                      chunk.tiles[tile].t_xmin,
                      chunk.tiles[tile].t_xmax,
                      chunk.tiles[tile].t_ymin,
                      chunk.tiles[tile].t_ymax,
                      g_big);
    f.compute(dt);
    *local_dt = dt;
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/calc_dt_kernel_c.c"
#include "../definitions_c.h"

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

#if defined(USE_CUDA)
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/calc_dt_kernel_c.c"
#include "../definitions_c.h"

__device__ unsigned long next_power_of_2(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

__global__ void calc_dt_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t xarea,
    const_field_2d_t yarea,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    const_field_2d_t volume,
    const_field_2d_t density0,
    const_field_2d_t energy0 ,
    const_field_2d_t pressure,
    const_field_2d_t viscosity,
    const_field_2d_t soundspeed,
    const_field_2d_t xvel0,
    const_field_2d_t yvel0,
    field_2d_t work_array1)
{
    extern __shared__ double sdata[];

    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    int x = threadIdx.x,
        y = threadIdx.y;
    int lid = x + blockDim.x * y;
    int gid = blockIdx.x + gridDim.x * blockIdx.y;

    int lsize = blockDim.x * blockDim.y;

    sdata[lid] = 1000.0;
    if (j <= x_max && k <= y_max) {
        double val = calc_dt_kernel_c_(
                         j, k,
                         x_min, x_max,
                         y_min, y_max,
                         xarea,
                         yarea,
                         celldx,
                         celldy,
                         volume,
                         density0,
                         energy0,
                         pressure,
                         viscosity,
                         soundspeed,
                         xvel0,
                         yvel0,
                         work_array1);
        sdata[lid] = val;
    }
    __syncthreads();
    for (int s = lsize / 2; s > 0; s /= 2) {
        if (lid < s) {
            if (sdata[lid + s] < sdata[lid])
                sdata[lid] = sdata[lid + s];
        }
        __syncthreads();
    }
    if (lid == 0) {
        work_array1[gid] = sdata[0];
    }
}

void calc_dt_adaptor(int tile, double* local_dt)
{
    double dt = g_big;
    int x_min = chunk.tiles[tile].t_xmin,
        x_max = chunk.tiles[tile].t_xmax,
        y_min = chunk.tiles[tile].t_ymin,
        y_max = chunk.tiles[tile].t_ymax;
    dim3 size = numBlocks(
                    dim3((x_max) - (x_min) + 1,
                         (y_max) - (y_min) + 1),
                    dtmin_blocksize);
    calc_dt_kernel <<< size,
                   dtmin_blocksize,
                   dtmin_blocksize.x* dtmin_blocksize.y*
                   sizeof(double)>>>(
                       x_min, x_max,
                       y_min, y_max,
                       chunk.tiles[tile].field.d_xarea,
                       chunk.tiles[tile].field.d_yarea,
                       chunk.tiles[tile].field.d_celldx,
                       chunk.tiles[tile].field.d_celldy,
                       chunk.tiles[tile].field.d_volume,
                       chunk.tiles[tile].field.d_density0,
                       chunk.tiles[tile].field.d_energy0,
                       chunk.tiles[tile].field.d_pressure,
                       chunk.tiles[tile].field.d_viscosity,
                       chunk.tiles[tile].field.d_soundspeed,
                       chunk.tiles[tile].field.d_xvel0,
                       chunk.tiles[tile].field.d_yvel0,
                       chunk.tiles[tile].field.d_work_array1);

    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.work_array1,
                  chunk.tiles[tile].field.d_work_array1,
                  size.x * size.y * sizeof(double),
                  cudaMemcpyDeviceToHost));
    for (int i = 0; i < size.x * size.y; i++) {
        double val = chunk.tiles[tile].field.work_array1[i];
        if (val < dt) {
            dt = val;
        }
    }
    *local_dt = dt;
    if (profiler_on)
        cudaDeviceSynchronize();
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

        checkOclErr(calc_dt.setArg(0,  x_min));
        checkOclErr(calc_dt.setArg(1,  x_max));
        checkOclErr(calc_dt.setArg(2,  y_min));
        checkOclErr(calc_dt.setArg(3,  y_max));

        checkOclErr(calc_dt.setArg(4, *chunk.tiles[tile].field.d_xarea));
        checkOclErr(calc_dt.setArg(5, *chunk.tiles[tile].field.d_yarea));
        checkOclErr(calc_dt.setArg(6, *chunk.tiles[tile].field.d_celldx));
        checkOclErr(calc_dt.setArg(7, *chunk.tiles[tile].field.d_celldy));
        checkOclErr(calc_dt.setArg(8, *chunk.tiles[tile].field.d_volume));
        checkOclErr(calc_dt.setArg(9, *chunk.tiles[tile].field.d_density0));
        checkOclErr(calc_dt.setArg(10, *chunk.tiles[tile].field.d_energy0));
        checkOclErr(calc_dt.setArg(11, *chunk.tiles[tile].field.d_pressure));
        checkOclErr(calc_dt.setArg(12, *chunk.tiles[tile].field.d_viscosity));
        checkOclErr(calc_dt.setArg(13, *chunk.tiles[tile].field.d_soundspeed));
        checkOclErr(calc_dt.setArg(14, *chunk.tiles[tile].field.d_xvel0));
        checkOclErr(calc_dt.setArg(15, *chunk.tiles[tile].field.d_yvel0));
        checkOclErr(calc_dt.setArg(16, *chunk.tiles[tile].field.d_work_array1));
        checkOclErr(calc_dt.setArg(17, sizeof(double) *
                                   dtmin_local_size[0] *
                                   dtmin_local_size[1], NULL));
        cl::NDRange global_size = calcGlobalSize(
                                      cl::NDRange(x_max - x_min + 1, y_max - y_min + 1),
                                      dtmin_local_size);
        checkOclErr(openclQueue.enqueueNDRangeKernel(
                        calc_dt, cl::NullRange,
                        global_size,
                        dtmin_local_size));

        openclQueue.finish();
        int num_groups = (global_size[0] / dtmin_local_size[0]) *
                         (global_size[1] / dtmin_local_size[1]);
        mapoclmem(chunk.tiles[tile].field.d_work_array1,
                  chunk.tiles[tile].field.work_array1,
                  num_groups,
                  CL_MAP_READ);

        for (int i = 0; i < num_groups; i++) {
            double val = chunk.tiles[tile].field.work_array1[i];
            if (val < min)
                min = val;
        }

        unmapoclmem(chunk.tiles[tile].field.d_work_array1,
                    chunk.tiles[tile].field.work_array1);
    }

    if (profiler_on)
        openclQueue.finish();
    *local_dt = min;
}
#endif
