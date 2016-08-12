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
                      chunk.tiles[tile].t_ymax);
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

__global__ void calc_dt_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double* xarea,
    double* yarea,
    double* celldx,
    double* celldy,
    double* volume,
    double* density0,
    double* energy0,
    double* pressure,
    double* viscosity,
    double* soundspeed,
    double* xvel0,
    double* yvel0,
    double* work_array1)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

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
        WORK_ARRAY(work_array1, j, k) = val;
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
    calc_dt_kernel <<< size, dtmin_blocksize>>>(
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
                  chunk.tiles[tile].field.work_array1_size * sizeof(double),
                  cudaMemcpyDeviceToHost));
    int jmin = -1, kmin = -1;
    for (int k = chunk.tiles[tile].t_ymin; k <= chunk.tiles[tile].t_ymax; k++) {
        for (int j = chunk.tiles[tile].t_xmin; j <= chunk.tiles[tile].t_xmax; j++) {
            double val = WORK_ARRAY(chunk.tiles[tile].field.work_array1, j, k);
            if (val < dt) {
                dt = val;
                jmin = j;
                kmin = k;
            }
        }
    }
    *local_dt = dt;
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
        // checkOclErr(calc_dt.setArg(17, sizeof(double) * 10 * 10, NULL));

        checkOclErr(openclQueue.enqueueNDRangeKernel(
                        calc_dt, cl::NullRange,
                        cl::NDRange(x_max - x_min + 1, y_max - y_min + 1),
                        dtmin_local_size));

        // cl::NDRange reductionLocalSize(1, 1);

        // cl::Kernel reduce(openclProgram, "reduce");
        // checkOclErr(reduce.setArg(0, *chunk.tiles[tile].field.d_work_array1));
        // checkOclErr(reduce.setArg(1, sizeof(double) * 960 * 960, NULL));
        // checkOclErr(reduce.setArg(2, 960 * 960));
        // checkOclErr(reduce.setArg(3, *chunk.tiles[tile].field.d_work_array2));

        // checkOclErr(openclQueue.enqueueNDRangeKernel(
        //                 reduce,
        //                 cl::NullRange,
        //                 cl::NDRange(100, 100),
        //                 reductionLocalSize));

        openclQueue.finish();

        mapoclmem(chunk.tiles[tile].field.d_work_array1,
                  chunk.tiles[tile].field.work_array1,
                  chunk.tiles[tile].field.work_array1_size,
                  CL_MAP_READ);

        for (int k = y_min; k <= y_max; k++) {
            for (int j = x_min; j <= x_max; j++) {
                double val = WORK_ARRAY(chunk.tiles[tile].field.work_array1, j, k);
                if (val < min)
                    min = val;
            }
        }
        // for (int i = 0; i < reductionLocalSize[0]*reductionLocalSize[1]; i++) {
        //     double val = chunk.tiles[tile].field.work_array2[i];
        //     if (val < min)
        //         min = val;
        // }

        unmapoclmem(chunk.tiles[tile].field.d_work_array1,
                    chunk.tiles[tile].field.work_array1);
    }

    if (profiler_on)
        openclQueue.finish();
    *local_dt = min;
}
#endif
