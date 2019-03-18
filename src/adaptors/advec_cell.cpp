#include "../definitions_c.h"

#ifdef USE_KOKKOS

#include "kokkos/advec_cell.cpp"

void advec_cell(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
    int g_xdir = 1, g_ydir = 2;

    if (dir == g_xdir) {
        xsweep_functor f1(
            tile,
            x_min - 2, x_max + 2, y_min - 2, y_max + 2,
            sweep_number);
        f1.compute();

        xcomp1_functor f2(
            tile,
            x_min, x_max + 2, y_min, y_max);
        f2.compute();

        xcomp2_functor f3(
            tile,
            x_min, x_max, y_min, y_max);
        f3.compute();
        Kokkos::fence();
    }
    if (dir == g_ydir) {
        ysweep_functor f1(
            tile,
            x_min - 2, x_max + 2, y_min - 2, y_max + 2,
            sweep_number);
        f1.compute();

        ycomp1_functor f2(
            tile,
            x_min, x_max, y_min, y_max + 2);
        f2.compute();

        ycomp2_functor f3(
            tile,
            x_min, x_max, y_min, y_max);
        f3.compute();
        Kokkos::fence();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/advec_cell_kernel_c.c"

void advec_cell(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
    const_field_1d_t vertexdx = tile.field.vertexdx;
    const_field_1d_t vertexdy = tile.field.vertexdy;
    const_field_2d_t volume = tile.field.volume;
    field_2d_t       density1 = tile.field.density1;
    field_2d_t       energy1 = tile.field.energy1;
    field_2d_t       mass_flux_x = tile.field.mass_flux_x;
    const_field_2d_t vol_flux_x = tile.field.vol_flux_x;
    field_2d_t       mass_flux_y = tile.field.mass_flux_y;
    const_field_2d_t vol_flux_y = tile.field.vol_flux_y;
    field_2d_t       pre_vol = tile.field.work_array1;
    field_2d_t       post_vol = tile.field.work_array2;
    field_2d_t       pre_mass = tile.field.work_array3;
    field_2d_t       post_mass = tile.field.work_array4;
    field_2d_t       advec_vol = tile.field.work_array5;
    field_2d_t       post_ener = tile.field.work_array6;
    field_2d_t       ener_flux = tile.field.work_array7;

    int g_xdir = 1, g_ydir = 2;

    #pragma omp parallel
    {
        if (dir == g_xdir) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                xsweep(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_vol, post_vol, volume, vol_flux_x, vol_flux_y,
                    sweep_number
                );
            });
            DOUBLEFOR(y_min, y_max, x_min, x_max + 2, {
                xcomp1(
                    j,  k,
                    x_min, x_max, y_min, y_max,
                    mass_flux_x, ener_flux, vol_flux_x,
                    pre_vol, density1, energy1, vertexdx
                );
            });

            DOUBLEFOR(y_min, y_max, x_min, x_max, {
                xcomp2(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_mass, post_mass, post_ener, advec_vol,
                    density1, energy1, pre_vol, mass_flux_x,
                    ener_flux, vol_flux_x
                );
            });
        }
        if (dir == g_ydir) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                ysweep(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_vol, post_vol, volume, vol_flux_x, vol_flux_y,
                    sweep_number
                );
            });
            DOUBLEFOR(y_min, y_max + 2, x_min , x_max , {
                ycomp1(
                    j,  k,
                    x_min,  x_max, y_min,  y_max,
                    mass_flux_y, ener_flux, vol_flux_y, pre_vol,
                    density1, energy1, vertexdy
                );
            });
            DOUBLEFOR(y_min, y_max, x_min, x_max, {
                ycomp2(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_mass, post_mass, post_ener, advec_vol, density1,
                    energy1, pre_vol, mass_flux_y, ener_flux, vol_flux_y
                );
            });
        }
    }
}

#endif
#if defined(USE_CUDA)

#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/advec_cell_kernel_c.cc"


__global__ void xsweep_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_vol,
    field_2d_t post_vol,
    const_field_2d_t volume,
    const_field_2d_t vol_flux_x,
    const_field_2d_t vol_flux_y,
    int sweep_number)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        xsweep(
            j, k,
            x_min, x_max,
            y_min, y_max,
            pre_vol,
            post_vol,
            volume,
            vol_flux_x,
            vol_flux_y,
            sweep_number);

}

__global__ void ysweep_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_vol,
    field_2d_t post_vol,
    const_field_2d_t volume,
    const_field_2d_t vol_flux_x,
    const_field_2d_t vol_flux_y,
    int sweep_number)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ysweep(
            j, k,
            x_min, x_max,
            y_min, y_max,
            pre_vol,
            post_vol,
            volume,
            vol_flux_x,
            vol_flux_y,
            sweep_number);

}

__global__ void xcomp1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mass_flux_x,
    field_2d_t ener_flux,
    const_field_2d_t vol_flux_x,
    const_field_2d_t pre_vol,
    const_field_2d_t density1,
    const_field_2d_t energy1,
    const_field_1d_t vertexdx)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max + 2 && k <= y_max)
        xcomp1(
            j,  k,
            x_min, x_max, y_min, y_max,
            mass_flux_x,
            ener_flux,
            vol_flux_x,
            pre_vol,
            density1,
            energy1,
            vertexdx);
}

__global__ void ycomp1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mass_flux_x,
    field_2d_t ener_flux,
    const_field_2d_t vol_flux_x,
    const_field_2d_t pre_vol,
    const_field_2d_t density1,
    const_field_2d_t energy1,
    const_field_1d_t vertexdx)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max + 2)
        ycomp1(
            j,  k,
            x_min, x_max, y_min, y_max,
            mass_flux_x,
            ener_flux,
            vol_flux_x,
            pre_vol,
            density1,
            energy1,
            vertexdx);
}

__global__ void xcomp2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_mass,
    field_2d_t post_mass,
    field_2d_t post_ener,
    field_2d_t advec_vol,
    field_2d_t density1,
    field_2d_t energy1,
    const_field_2d_t pre_vol,
    const_field_2d_t mass_flux_x,
    const_field_2d_t ener_flux,
    const_field_2d_t vol_flux_x)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max)
        xcomp2(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_mass,
            post_mass,
            post_ener,
            advec_vol,
            density1,
            energy1,
            pre_vol,
            mass_flux_x,
            ener_flux,
            vol_flux_x);
}
__global__ void ycomp2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_mass,
    field_2d_t post_mass,
    field_2d_t post_ener,
    field_2d_t advec_vol,
    field_2d_t density1,
    field_2d_t energy1,
    const_field_2d_t pre_vol,
    const_field_2d_t mass_flux_x,
    const_field_2d_t ener_flux,
    const_field_2d_t vol_flux_x)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max + 2 && k <= y_max + 2)
        ycomp2(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_mass,
            post_mass,
            post_ener,
            advec_vol,
            density1,
            energy1,
            pre_vol,
            mass_flux_x,
            ener_flux,
            vol_flux_x);
}

void advec_cell(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
    int g_xdir = 1, g_ydir = 2;



    if (dir == g_xdir) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1,
                              (y_max + 2) - (y_min - 2) + 1),
                         advec_cell_x1_blocksize);
        xsweep_kernel <<< size1, advec_cell_x1_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array1,
            tile.field.d_work_array2,
            tile.field.d_volume,
            tile.field.d_vol_flux_x,
            tile.field.d_vol_flux_y,
            sweep_number);

        dim3 size2 = numBlocks(
                         dim3((x_max + 2) - (x_min) + 1,
                              (y_max) - (y_min) + 1),
                         advec_cell_x2_blocksize);
        xcomp1_kernel <<< size2, advec_cell_x2_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_mass_flux_x,
            tile.field.d_work_array7,
            tile.field.d_vol_flux_x,
            tile.field.d_work_array1,
            tile.field.d_density1,
            tile.field.d_energy1,
            tile.field.d_vertexdx);


        dim3 size3 = numBlocks(
                         dim3((x_max) - (x_min) + 1,
                              (y_max) - (y_min) + 1),
                         advec_cell_x3_blocksize);
        xcomp2_kernel <<< size3, advec_cell_x3_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array3,
            tile.field.d_work_array4,
            tile.field.d_work_array6,
            tile.field.d_work_array5,
            tile.field.d_density1,
            tile.field.d_energy1,
            tile.field.d_work_array1,
            tile.field.d_mass_flux_x,
            tile.field.d_work_array7,
            tile.field.d_vol_flux_x);

    }
    if (dir == g_ydir) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                         advec_cell_y1_blocksize);
        ysweep_kernel <<< size1, advec_cell_y1_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array1,
            tile.field.d_work_array2,
            tile.field.d_volume,
            tile.field.d_vol_flux_x,
            tile.field.d_vol_flux_y,
            sweep_number);

        dim3 size2 = numBlocks(
                         dim3((x_max) - (x_min) + 1,
                              (y_max + 2) - (y_min) + 1),
                         advec_cell_y2_blocksize);
        ycomp1_kernel <<< size2, advec_cell_y2_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_mass_flux_y,
            tile.field.d_work_array7,
            tile.field.d_vol_flux_y,
            tile.field.d_work_array1,
            tile.field.d_density1,
            tile.field.d_energy1,
            tile.field.d_vertexdy);


        dim3 size3 = numBlocks(
                         dim3((x_max) - (x_min) + 1, (y_max) - (y_min) + 1),
                         advec_cell_y3_blocksize);
        ycomp2_kernel <<< size3, advec_cell_y3_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array3,
            tile.field.d_work_array4,
            tile.field.d_work_array6,
            tile.field.d_work_array5,
            tile.field.d_density1,
            tile.field.d_energy1,
            tile.field.d_work_array1,
            tile.field.d_mass_flux_y,
            tile.field.d_work_array7,
            tile.field.d_vol_flux_y);
    }

    if (profiler_on)
        cudaDeviceSynchronize();
}

#endif

#if defined(USE_OPENCL)
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/advec_cell_kernel_c.c"
#include "../definitions_c.h"

void advec_cell(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
    int g_xdir = 1,
        g_ydir = 2;

    if (dir == g_xdir) {
        cl::Kernel xsweep_kernel(openclProgram, "xsweep_kernel");
        xsweep_kernel.setArg(0,  x_min);
        xsweep_kernel.setArg(1,  x_max);
        xsweep_kernel.setArg(2,  y_min);
        xsweep_kernel.setArg(3,  y_max);
        xsweep_kernel.setArg(4,  *tile.field.d_work_array1);
        xsweep_kernel.setArg(5,  *tile.field.d_work_array2);
        xsweep_kernel.setArg(6,  *tile.field.d_volume);
        xsweep_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        xsweep_kernel.setArg(8,  *tile.field.d_vol_flux_y);
        xsweep_kernel.setArg(9,  sweep_number);
        openclQueue.enqueueNDRangeKernel(
            xsweep_kernel,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_cell_x1_local_size),
            advec_cell_x1_local_size);

        cl::Kernel xcomp1_kernel(openclProgram, "xcomp1_kernel");
        xcomp1_kernel.setArg(0,  x_min);
        xcomp1_kernel.setArg(1,  x_max);
        xcomp1_kernel.setArg(2,  y_min);
        xcomp1_kernel.setArg(3,  y_max);
        xcomp1_kernel.setArg(4,  *tile.field.d_mass_flux_x);
        xcomp1_kernel.setArg(5,  *tile.field.d_work_array7);
        xcomp1_kernel.setArg(6,  *tile.field.d_vol_flux_x);
        xcomp1_kernel.setArg(7,  *tile.field.d_work_array1);
        xcomp1_kernel.setArg(8,  *tile.field.d_density1);
        xcomp1_kernel.setArg(9,  *tile.field.d_energy1);
        xcomp1_kernel.setArg(10, *tile.field.d_vertexdx);
        openclQueue.enqueueNDRangeKernel(
            xcomp1_kernel,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min) + 1, (y_max) - (y_min) + 1),
                           advec_cell_x2_local_size),
            advec_cell_x2_local_size);

        cl::Kernel xcomp2_kernel(openclProgram, "xcomp2_kernel");
        xcomp2_kernel.setArg(0,  x_min);
        xcomp2_kernel.setArg(1,  x_max);
        xcomp2_kernel.setArg(2,  y_min);
        xcomp2_kernel.setArg(3,  y_max);
        xcomp2_kernel.setArg(4,  *tile.field.d_work_array3);
        xcomp2_kernel.setArg(5,  *tile.field.d_work_array4);
        xcomp2_kernel.setArg(6,  *tile.field.d_work_array6);
        xcomp2_kernel.setArg(7,  *tile.field.d_work_array5);
        xcomp2_kernel.setArg(8,  *tile.field.d_density1);
        xcomp2_kernel.setArg(9,  *tile.field.d_energy1);
        xcomp2_kernel.setArg(10,  *tile.field.d_work_array1);
        xcomp2_kernel.setArg(11,  *tile.field.d_mass_flux_x);
        xcomp2_kernel.setArg(12,  *tile.field.d_work_array7);
        xcomp2_kernel.setArg(13,  *tile.field.d_vol_flux_x);
        openclQueue.enqueueNDRangeKernel(
            xcomp2_kernel,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max) - (x_min) + 1, (y_max) - (y_min) + 1),
                           advec_cell_x3_local_size),
            advec_cell_x3_local_size);
    }
    if (dir == g_ydir) {
        cl::Kernel ysweep_kernel(openclProgram, "ysweep_kernel");
        ysweep_kernel.setArg(0,  x_min);
        ysweep_kernel.setArg(1,  x_max);
        ysweep_kernel.setArg(2,  y_min);
        ysweep_kernel.setArg(3,  y_max);
        ysweep_kernel.setArg(4,  *tile.field.d_work_array1);
        ysweep_kernel.setArg(5,  *tile.field.d_work_array2);
        ysweep_kernel.setArg(6,  *tile.field.d_volume);
        ysweep_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ysweep_kernel.setArg(8,  *tile.field.d_vol_flux_y);
        ysweep_kernel.setArg(9,  sweep_number);
        openclQueue.enqueueNDRangeKernel(
            ysweep_kernel,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_cell_y1_local_size),
            advec_cell_y1_local_size);

        cl::Kernel ycomp1_kernel(openclProgram, "ycomp1_kernel");
        ycomp1_kernel.setArg(0,  x_min);
        ycomp1_kernel.setArg(1,  x_max);
        ycomp1_kernel.setArg(2,  y_min);
        ycomp1_kernel.setArg(3,  y_max);
        ycomp1_kernel.setArg(4,  *tile.field.d_mass_flux_y);
        ycomp1_kernel.setArg(5,  *tile.field.d_work_array7);
        ycomp1_kernel.setArg(6,  *tile.field.d_vol_flux_y);
        ycomp1_kernel.setArg(7,  *tile.field.d_work_array1);
        ycomp1_kernel.setArg(8,  *tile.field.d_density1);
        ycomp1_kernel.setArg(9,  *tile.field.d_energy1);
        ycomp1_kernel.setArg(10, *tile.field.d_vertexdy);
        openclQueue.enqueueNDRangeKernel(
            ycomp1_kernel,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max) - (x_min) + 1, (y_max + 2) - (y_min) + 1),
                           advec_cell_y2_local_size),
            advec_cell_y2_local_size);

        cl::Kernel ycomp2_kernel(openclProgram, "ycomp2_kernel");
        ycomp2_kernel.setArg(0,  x_min);
        ycomp2_kernel.setArg(1,  x_max);
        ycomp2_kernel.setArg(2,  y_min);
        ycomp2_kernel.setArg(3,  y_max);
        ycomp2_kernel.setArg(4,  *tile.field.d_work_array3);
        ycomp2_kernel.setArg(5,  *tile.field.d_work_array4);
        ycomp2_kernel.setArg(6,  *tile.field.d_work_array6);
        ycomp2_kernel.setArg(7,  *tile.field.d_work_array5);
        ycomp2_kernel.setArg(8,  *tile.field.d_density1);
        ycomp2_kernel.setArg(9,  *tile.field.d_energy1);
        ycomp2_kernel.setArg(10,  *tile.field.d_work_array1);
        ycomp2_kernel.setArg(11,  *tile.field.d_mass_flux_y);
        ycomp2_kernel.setArg(12,  *tile.field.d_work_array7);
        ycomp2_kernel.setArg(13,  *tile.field.d_vol_flux_y);
        openclQueue.enqueueNDRangeKernel(
            ycomp2_kernel,
            cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max) - (x_min) + 1, (y_max) - (y_min) + 1),
                           advec_cell_y3_local_size),
            advec_cell_y3_local_size);
    }
    if (profiler_on)
        openclQueue.finish();
}

#endif