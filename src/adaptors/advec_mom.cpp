#include "../definitions_c.h"

#define X 1
#define Y 2

#if defined(USE_KOKKOS)
#include "kokkos/mom_sweep.cpp"
#include "kokkos/mom_direction.cpp"


void advec_mom(
    int which_vel,
    struct tile_type tile,
    int x_min, int x_max, int y_min, int y_max,
    int sweep_number,
    int direction)
{
    auto vel1 = which_vel == 1 ? tile.field.d_xvel1 : tile.field.d_yvel1;
    int mom_sweep = direction + 2 * (sweep_number - 1);

    mom_sweep_functor f1(
        tile,
        x_min - 2, x_max + 2, y_min - 2, y_max + 2,
        mom_sweep);
    f1.compute();

    if (direction == X) {
        mom_direction_x1_functor f2(
            tile,
            x_min - 2, x_max + 2, y_min, y_max + 1);
        f2.compute();
    } else if (direction == Y) {
        mom_direction_y1_functor f2(
            tile,
            x_min, x_max + 1, y_min - 2, y_max + 2);
        f2.compute();
    }

    if (direction == X) {
        mom_direction_x2_functor f3(
            tile,
            x_min - 1, x_max + 2, y_min , y_max + 1);
        f3.compute();

    } else if (direction == Y) {
        mom_direction_y2_functor f3(
            tile,
            x_min , x_max + 1, y_min - 1 , y_max + 2);
        f3.compute();
    }

    if (direction == X) {
        mom_direction_x3_functor f4(
            tile,
            x_min, x_max + 1, y_min , y_max + 1,
            vel1);
        f4.compute();
    } else if (direction == Y) {
        mom_direction_y3_functor f4(
            tile,
            x_min, x_max + 1, y_min, y_max + 1,
            vel1);
        f4.compute();
    }

    if (direction == X) {
        mom_direction_x4_functor f4(
            tile,
            x_min, x_max + 1, y_min , y_max + 1,
            vel1);
        f4.compute();
    } else if (direction == Y) {
        mom_direction_y4_functor f4(
            tile,
            x_min, x_max + 1, y_min , y_max + 1,
            vel1);
        f4.compute();
    }
}

#endif


#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/advec_mom_kernel_c.cc"

void advec_mom(
    int which_vel,
    struct tile_type tile,
    int x_min, int x_max, int y_min, int y_max,
    int sweep_number,
    int direction)
{
    const_field_2d_t mass_flux_x    = tile.field.mass_flux_x;
    const_field_2d_t vol_flux_x     = tile.field.vol_flux_x;
    const_field_2d_t mass_flux_y    = tile.field.mass_flux_y;
    const_field_2d_t vol_flux_y     = tile.field.vol_flux_y;
    const_field_2d_t volume         = tile.field.volume;
    const_field_2d_t density1       = tile.field.density1;
    field_2d_t       node_flux      = tile.field.work_array1;
    field_2d_t       node_mass_post = tile.field.work_array2;
    field_2d_t       node_mass_pre  = tile.field.work_array3;
    field_2d_t       mom_flux       = tile.field.work_array4;
    field_2d_t       pre_vol        = tile.field.work_array5;
    field_2d_t       post_vol       = tile.field.work_array6;
    const_field_1d_t celldx         = tile.field.celldx;
    const_field_1d_t celldy         = tile.field.celldy;
    field_2d_t vel1 = which_vel == 1 ? tile.field.xvel1 : tile.field.yvel1;
    int mom_sweep = direction + 2 * (sweep_number - 1);
    #pragma omp parallel
    {
        if (mom_sweep == 1) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                ms1(j, k, x_min, x_max, y_min, y_max, pre_vol, post_vol, volume, vol_flux_x, vol_flux_y);
            });
        } else if (mom_sweep == 2) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                ms2(j, k, x_min, x_max, y_min, y_max, pre_vol, post_vol, volume, vol_flux_x, vol_flux_y);
            });
        } else if (mom_sweep == 3) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                ms3(j, k, x_min, x_max, y_min, y_max, pre_vol, post_vol, volume, vol_flux_x, vol_flux_y);
            });
        } else if (mom_sweep == 4) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                ms4(j, k, x_min, x_max, y_min, y_max, pre_vol, post_vol, volume, vol_flux_x, vol_flux_y);
            });
        }

        if (direction == X) {
            DOUBLEFOR(y_min, y_max + 1, x_min - 2, x_max + 2, {
                dx1(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    node_flux,
                    mass_flux_x);
            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 2, {
                dx2(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    node_mass_post,
                    node_mass_pre,
                    density1,
                    post_vol,
                    node_flux);
            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 1, {
                dx3(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    mom_flux,
                    node_flux,
                    node_mass_pre,
                    celldx,
                    vel1);
            });

            DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, {
                dx4(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    vel1,
                    node_mass_pre,
                    mom_flux,
                    node_mass_post);

            });
        } else if (direction == Y) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min , x_max + 1, {

                dy1(
                    j,  k,
                    x_min,  x_max,  y_min,  y_max,
                    node_flux,
                    mass_flux_y);

            });

            DOUBLEFOR(y_min - 1, y_max + 2, x_min, x_max + 1, {
                dy2(
                    j,  k,
                    x_min,  x_max,  y_min,  y_max,
                    node_mass_post,
                    node_mass_pre,
                    density1,
                    post_vol,
                    node_flux);
            });

            DOUBLEFOR(y_min - 1, y_max + 1, x_min , x_max + 1, {
                dy3(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    mom_flux,
                    node_flux,
                    node_mass_pre,
                    celldy,
                    vel1);
            });

            DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, {
                dy4(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    vel1,
                    node_mass_pre,
                    mom_flux,
                    node_mass_post);
            });
        }
    }
}
#endif

#if defined(USE_CUDA)
#include <math.h>
#include "../kernels/ftocmacros.h"
#include "../kernels/advec_mom_kernel_c.cc"

__global__ void ms1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms1(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol ,
            volume ,
            vol_flux_x ,
            vol_flux_y);
}

__global__ void ms2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms2(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol ,
            volume ,
            vol_flux_x ,
            vol_flux_y);
}

__global__ void ms3_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms3(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol ,
            volume ,
            vol_flux_x ,
            vol_flux_y);
}

__global__ void ms4_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms4(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol ,
            volume ,
            vol_flux_x ,
            vol_flux_y);
}

__global__ void dx1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_flux,
    const_field_2d_t mass_flux_x)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 2;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max + 2 && k <= y_max + 1)
        dx1(
            j, k,
            x_min, x_max, y_min, y_max,
            node_flux,
            mass_flux_x);
}

__global__ void dy1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_flux,
    const_field_2d_t mass_flux_x)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 2;

    if (j <= x_max + 1 && k <= y_max + 2)
        dy1(
            j, k,
            x_min, x_max, y_min, y_max,
            node_flux,
            mass_flux_x);
}

__global__ void dx2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_mass_post,
    field_2d_t node_mass_pre,
    const_field_2d_t density1,
    const_field_2d_t post_vol,
    const_field_2d_t node_flux)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 1;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max + 1 && k <= y_max + 1)
        dx2(
            j, k,
            x_min, x_max, y_min, y_max,
            node_mass_post,
            node_mass_pre,
            density1,
            post_vol,
            node_flux);
}
__global__ void dy2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_mass_post,
    field_2d_t node_mass_pre,
    const_field_2d_t density1,
    const_field_2d_t post_vol,
    const_field_2d_t node_flux)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 1;

    if (j <= x_max + 1 && k <= y_max + 1)
        dy2(
            j, k,
            x_min, x_max, y_min, y_max,
            node_mass_post,
            node_mass_pre,
            density1,
            post_vol,
            node_flux);
}

__global__ void dx3_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mom_flux,
    const_field_2d_t node_flux,
    const_field_2d_t node_mass_pre,
    const_field_1d_t celldx,
    const_field_2d_t vel1)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min - 1;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max + 1 && k <= y_max + 1)
        dx3(
            j, k,
            x_min, x_max, y_min, y_max,
            mom_flux,
            node_flux,
            node_mass_pre,
            celldx,
            vel1);
}

__global__ void dy3_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mom_flux,
    const_field_2d_t node_flux,
    const_field_2d_t node_mass_pre,
    const_field_1d_t celldx,
    const_field_2d_t vel1)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min - 1;

    if (j <= x_max + 1 && k <= y_max + 1)
        dy3(
            j, k,
            x_min, x_max, y_min, y_max,
            mom_flux,
            node_flux,
            node_mass_pre,
            celldx,
            vel1);
}

__global__ void dx4_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t node_mass_pre,
    const_field_2d_t mom_flux,
    const_field_2d_t node_mass_post)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max && k <= y_max)
        dx4(
            j, k,
            x_min, x_max, y_min, y_max,
            vel1,
            node_mass_pre,
            mom_flux,
            node_mass_post);
}

__global__ void dy4_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t node_mass_pre,
    const_field_2d_t mom_flux,
    const_field_2d_t node_mass_post)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    if (j <= x_max  && k <= y_max)
        dy4(
            j, k,
            x_min, x_max, y_min, y_max,
            vel1,
            node_mass_pre,
            mom_flux,
            node_mass_post);
}

void advec_mom(
    int which_vel,
    struct tile_type tile,
    int x_min, int x_max, int y_min, int y_max,
    int sweep_number,
    int direction)
{
    double* d_vel1 = which_vel == 1 ? tile.field.d_xvel1 : tile.field.d_yvel1;

    int mom_sweep = direction + 2 * (sweep_number - 1);

    if (mom_sweep == 1) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1,
                              (y_max + 2) - (y_min - 2) + 1),
                         advec_mom_ms1_blocksize);
        ms1_kernel <<< size1, advec_mom_ms1_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array5,
            tile.field.d_work_array6,
            tile.field.d_volume,
            tile.field.d_vol_flux_x,
            tile.field.d_vol_flux_y);

    } else if (mom_sweep == 2) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1,
                              (y_max + 2) - (y_min - 2) + 1),
                         advec_mom_ms2_blocksize);
        ms2_kernel <<< size1, advec_mom_ms2_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array5,
            tile.field.d_work_array6,
            tile.field.d_volume,
            tile.field.d_vol_flux_x,
            tile.field.d_vol_flux_y);
    } else if (mom_sweep == 3) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1,
                              (y_max + 2) - (y_min - 2) + 1),
                         advec_mom_ms3_blocksize);
        ms3_kernel <<< size1, advec_mom_ms3_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array5,
            tile.field.d_work_array6,
            tile.field.d_volume,
            tile.field.d_vol_flux_x,
            tile.field.d_vol_flux_y);
    } else if (mom_sweep == 4) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1,
                              (y_max + 2) - (y_min - 2) + 1),
                         advec_mom_ms4_blocksize);
        ms4_kernel <<< size1, advec_mom_ms4_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array5,
            tile.field.d_work_array6,
            tile.field.d_volume,
            tile.field.d_vol_flux_x,
            tile.field.d_vol_flux_y);
    }

    if (direction == X) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 2) - (x_min - 2) + 1,
                              (y_max + 1) - (y_min) + 1),
                         advec_mom_x1_blocksize);
        dx1_kernel <<< size1, advec_mom_x1_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array1,
            tile.field.d_mass_flux_x);

        dim3 size2 = numBlocks(
                         dim3((x_max + 2) - (x_min - 1) + 1,
                              (y_max + 1) - (y_min) + 1),
                         advec_mom_x2_blocksize);
        dx2_kernel <<< size2, advec_mom_x2_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array2,
            tile.field.d_work_array3,
            tile.field.d_density1,
            tile.field.d_work_array6,
            tile.field.d_work_array1);

        dim3 size3 = numBlocks(
                         dim3((x_max + 1) - (x_min - 1) + 1,
                              (y_max + 1) - (y_min) + 1),
                         advec_mom_x3_blocksize);
        dx3_kernel <<< size3, advec_mom_x3_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array4,
            tile.field.d_work_array1,
            tile.field.d_work_array3,
            tile.field.d_celldx,
            d_vel1);

        dim3 size4 = numBlocks(
                         dim3((x_max + 1) - (x_min) + 1,
                              (y_max + 1) - (y_min) + 1),
                         advec_mom_x4_blocksize);
        dx4_kernel <<< size4, advec_mom_x4_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            d_vel1,
            tile.field.d_work_array3,
            tile.field.d_work_array4,
            tile.field.d_work_array2);

    } else if (direction == Y) {
        dim3 size1 = numBlocks(
                         dim3((x_max + 1) - (x_min) + 1,
                              (y_max + 2) - (y_min - 2) + 1),
                         advec_mom_y1_blocksize);
        dy1_kernel <<< size1, advec_mom_y1_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array1,
            tile.field.d_mass_flux_y);

        dim3 size2 = numBlocks(
                         dim3((x_max + 1) - (x_min) + 1,
                              (y_max + 2) - (y_min - 1) + 1),
                         advec_mom_y2_blocksize);
        dy2_kernel <<< size2, advec_mom_y2_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array2,
            tile.field.d_work_array3,
            tile.field.d_density1,
            tile.field.d_work_array6,
            tile.field.d_work_array1);

        dim3 size3 = numBlocks(
                         dim3((x_max + 1) - (x_min) + 1,
                              (y_max + 1) - (y_min - 1) + 1),
                         advec_mom_y3_blocksize);
        dy3_kernel <<< size3, advec_mom_y3_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            tile.field.d_work_array4,
            tile.field.d_work_array1,
            tile.field.d_work_array3,
            tile.field.d_celldy,
            d_vel1);

        dim3 size4 = numBlocks(
                         dim3((x_max + 1) - (x_min) + 1,
                              (y_max + 1) - (y_min) + 1),
                         advec_mom_y4_blocksize);
        dy4_kernel <<< size4, advec_mom_y4_blocksize >>> (
            x_min, x_max,
            y_min, y_max,
            d_vel1,
            tile.field.d_work_array3,
            tile.field.d_work_array4,
            tile.field.d_work_array2);

    }

    if (profiler_on)
        cudaDeviceSynchronize();
}
#endif


#if defined(USE_OPENCL)

#include "../kernels/advec_mom_kernel_c.cc"
// #include "../cl.hpp"
#include "../definitions_c.h"

void advec_mom(
    int which_vel,
    struct tile_type tile,
    int x_min, int x_max, int y_min, int y_max,
    int sweep_number,
    int direction)
{
    cl::Buffer* d_vel1 = which_vel == 1 ? tile.field.d_xvel1 : tile.field.d_yvel1;
    int mom_sweep = direction + 2 * (sweep_number - 1);

    if (mom_sweep == 1) {
        cl::Kernel ms1_kernel(openclProgram, "ms1_kernel");
        ms1_kernel.setArg(0,  x_min);
        ms1_kernel.setArg(1,  x_max);
        ms1_kernel.setArg(2,  y_min);
        ms1_kernel.setArg(3,  y_max);
        ms1_kernel.setArg(4,  *tile.field.d_work_array5);
        ms1_kernel.setArg(5,  *tile.field.d_work_array6);
        ms1_kernel.setArg(6,  *tile.field.d_volume);
        ms1_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms1_kernel.setArg(8,  *tile.field.d_vol_flux_y);
        openclQueue.enqueueNDRangeKernel(
            ms1_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_mom_ms1_local_size),
            advec_mom_ms1_local_size);
    } else if (mom_sweep == 2) {
        cl::Kernel ms2_kernel(openclProgram, "ms2_kernel");
        ms2_kernel.setArg(0,  x_min);
        ms2_kernel.setArg(1,  x_max);
        ms2_kernel.setArg(2,  y_min);
        ms2_kernel.setArg(3,  y_max);
        ms2_kernel.setArg(4,  *tile.field.d_work_array5);
        ms2_kernel.setArg(5,  *tile.field.d_work_array6);
        ms2_kernel.setArg(6,  *tile.field.d_volume);
        ms2_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms2_kernel.setArg(8,  *tile.field.d_vol_flux_y);
        openclQueue.enqueueNDRangeKernel(
            ms2_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_mom_ms2_local_size),
            advec_mom_ms2_local_size);
    } else if (mom_sweep == 3) {
        cl::Kernel ms3_kernel(openclProgram, "ms3_kernel");
        ms3_kernel.setArg(0,  x_min);
        ms3_kernel.setArg(1,  x_max);
        ms3_kernel.setArg(2,  y_min);
        ms3_kernel.setArg(3,  y_max);
        ms3_kernel.setArg(4,  *tile.field.d_work_array5);
        ms3_kernel.setArg(5,  *tile.field.d_work_array6);
        ms3_kernel.setArg(6,  *tile.field.d_volume);
        ms3_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms3_kernel.setArg(8,  *tile.field.d_vol_flux_y);
        openclQueue.enqueueNDRangeKernel(
            ms3_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_mom_ms3_local_size),
            advec_mom_ms3_local_size);
    } else if (mom_sweep == 4) {
        cl::Kernel ms4_kernel(openclProgram, "ms4_kernel");

        ms4_kernel.setArg(0,  x_min);
        ms4_kernel.setArg(1,  x_max);
        ms4_kernel.setArg(2,  y_min);
        ms4_kernel.setArg(3,  y_max);
        ms4_kernel.setArg(4,  *tile.field.d_work_array5);
        ms4_kernel.setArg(5,  *tile.field.d_work_array6);
        ms4_kernel.setArg(6,  *tile.field.d_volume);
        ms4_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms4_kernel.setArg(8,  *tile.field.d_vol_flux_y);

        openclQueue.enqueueNDRangeKernel(
            ms4_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_mom_ms4_local_size),
            advec_mom_ms4_local_size);
    }

    if (direction == X) {
        cl::Kernel dx1_kernel(openclProgram, "dx1_kernel");
        dx1_kernel.setArg(0,  x_min);
        dx1_kernel.setArg(1,  x_max);
        dx1_kernel.setArg(2,  y_min);
        dx1_kernel.setArg(3,  y_max);
        dx1_kernel.setArg(4,  *tile.field.d_work_array1);
        dx1_kernel.setArg(5,  *tile.field.d_mass_flux_x);
        openclQueue.enqueueNDRangeKernel(
            dx1_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 2) + 1, (y_max + 1) - (y_min) + 1),
                           advec_mom_x1_local_size),
            advec_mom_x1_local_size);

        cl::Kernel dx2_kernel(openclProgram, "dx2_kernel");
        dx2_kernel.setArg(0,  x_min);
        dx2_kernel.setArg(1,  x_max);
        dx2_kernel.setArg(2,  y_min);
        dx2_kernel.setArg(3,  y_max);
        dx2_kernel.setArg(4,  *tile.field.d_work_array2);
        dx2_kernel.setArg(5,  *tile.field.d_work_array3);
        dx2_kernel.setArg(6,  *tile.field.d_density1);
        dx2_kernel.setArg(7,  *tile.field.d_work_array6);
        dx2_kernel.setArg(8,  *tile.field.d_work_array1);
        openclQueue.enqueueNDRangeKernel(
            dx2_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 2) - (x_min - 1) + 1, (y_max + 1) - (y_min) + 1),
                           advec_mom_x2_local_size),
            advec_mom_x2_local_size);

        cl::Kernel dx3_kernel(openclProgram, "dx3_kernel");
        dx3_kernel.setArg(0,  x_min);
        dx3_kernel.setArg(1,  x_max);
        dx3_kernel.setArg(2,  y_min);
        dx3_kernel.setArg(3,  y_max);
        dx3_kernel.setArg(4,  *tile.field.d_work_array4);
        dx3_kernel.setArg(5,  *tile.field.d_work_array1);
        dx3_kernel.setArg(6,  *tile.field.d_work_array3);
        dx3_kernel.setArg(7,  *tile.field.d_celldx);
        dx3_kernel.setArg(8,  *d_vel1);
        openclQueue.enqueueNDRangeKernel(
            dx3_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 1) - (x_min - 1) + 1, (y_max + 1) - (y_min) + 1),
                           advec_mom_x3_local_size),
            advec_mom_x3_local_size);

        cl::Kernel dx4_kernel(openclProgram, "dx4_kernel");
        dx4_kernel.setArg(0,  x_min);
        dx4_kernel.setArg(1,  x_max);
        dx4_kernel.setArg(2,  y_min);
        dx4_kernel.setArg(3,  y_max);
        dx4_kernel.setArg(4,  *d_vel1);
        dx4_kernel.setArg(5,  *tile.field.d_work_array3);
        dx4_kernel.setArg(6,  *tile.field.d_work_array4);
        dx4_kernel.setArg(7,  *tile.field.d_work_array2);
        openclQueue.enqueueNDRangeKernel(
            dx4_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 1) - (x_min) + 1, (y_max + 1) - (y_min) + 1),
                           advec_mom_x4_local_size),
            advec_mom_x4_local_size);
    } else if (direction == Y) {
        cl::Kernel dy1_kernel(openclProgram, "dy1_kernel");
        dy1_kernel.setArg(0,  x_min);
        dy1_kernel.setArg(1,  x_max);
        dy1_kernel.setArg(2,  y_min);
        dy1_kernel.setArg(3,  y_max);
        dy1_kernel.setArg(4,  *tile.field.d_work_array1);
        dy1_kernel.setArg(5,  *tile.field.d_mass_flux_y);
        openclQueue.enqueueNDRangeKernel(
            dy1_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 1) - (x_min) + 1, (y_max + 2) - (y_min - 2) + 1),
                           advec_mom_y1_local_size),
            advec_mom_y1_local_size);

        cl::Kernel dy2_kernel(openclProgram, "dy2_kernel");
        dy2_kernel.setArg(0,  x_min);
        dy2_kernel.setArg(1,  x_max);
        dy2_kernel.setArg(2,  y_min);
        dy2_kernel.setArg(3,  y_max);
        dy2_kernel.setArg(4,  *tile.field.d_work_array2);
        dy2_kernel.setArg(5,  *tile.field.d_work_array3);
        dy2_kernel.setArg(6,  *tile.field.d_density1);
        dy2_kernel.setArg(7,  *tile.field.d_work_array6);
        dy2_kernel.setArg(8,  *tile.field.d_work_array1);
        openclQueue.enqueueNDRangeKernel(
            dy2_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 1) - (x_min) + 1, (y_max + 2) - (y_min - 1) + 1),
                           advec_mom_y2_local_size),
            advec_mom_y2_local_size);

        cl::Kernel dy3_kernel(openclProgram, "dy3_kernel");
        dy3_kernel.setArg(0,  x_min);
        dy3_kernel.setArg(1,  x_max);
        dy3_kernel.setArg(2,  y_min);
        dy3_kernel.setArg(3,  y_max);
        dy3_kernel.setArg(4,  *tile.field.d_work_array4);
        dy3_kernel.setArg(5,  *tile.field.d_work_array1);
        dy3_kernel.setArg(6,  *tile.field.d_work_array3);
        dy3_kernel.setArg(7,  *tile.field.d_celldy);
        dy3_kernel.setArg(8,  *d_vel1);
        openclQueue.enqueueNDRangeKernel(
            dy3_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 1) - (x_min) + 1, (y_max + 1) - (y_min - 1) + 1),
                           advec_mom_y3_local_size),
            advec_mom_y3_local_size);

        cl::Kernel dy4_kernel(openclProgram, "dy4_kernel");
        dy4_kernel.setArg(0,  x_min);
        dy4_kernel.setArg(1,  x_max);
        dy4_kernel.setArg(2,  y_min);
        dy4_kernel.setArg(3,  y_max);
        dy4_kernel.setArg(4,  *d_vel1);
        dy4_kernel.setArg(5,  *tile.field.d_work_array3);
        dy4_kernel.setArg(6,  *tile.field.d_work_array4);
        dy4_kernel.setArg(7,  *tile.field.d_work_array2);
        openclQueue.enqueueNDRangeKernel(
            dy4_kernel, cl::NullRange,
            calcGlobalSize(cl::NDRange((x_max + 1) - (x_min) + 1, (y_max + 1) - (y_min) + 1),
                           advec_mom_y4_local_size),
            advec_mom_y4_local_size);
    }
    if (profiler_on)
        openclQueue.finish();
}
#endif
