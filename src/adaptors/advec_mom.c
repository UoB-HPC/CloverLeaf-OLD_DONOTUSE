#include "../definitions_c.h"

#define X 1
#define Y 2

#if defined(USE_KOKKOS)
#include "kokkos/mom_sweep.cpp"
#include "kokkos/mom_direction.cpp"

void advec_mom(
    field_2d_t vel1,
    struct tile_type tile,
    int x_min, int x_max, int y_min, int y_max,
    int sweep_number,
    int direction)
{
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
            x_min - 1, x_max + 1, y_min - 1 , y_max + 1,
            vel1);
        f4.compute();
    } else if (direction == Y) {
        mom_direction_y3_functor f4(
            tile,
            x_min, x_max + 1, y_min - 1 , y_max + 1,
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
#include "../kernels/advec_mom_kernel_c.c"

void advec_mom(
    field_2d_t vel1,
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

        if (direction == 1) {
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
        } else if (direction == 2) {
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


#if defined(USE_OPENCL)
#include "../kernels/advec_mom_kernel_c.c"
void advec_mom(
    field_2d_t vel1,
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
    int mom_sweep = direction + 2 * (sweep_number - 1);

    if (mom_sweep == 1) {
        cl::Kernel ms1_kernel(openclProgram, "ms1_kernel");

        int xmin = tile.t_xmin,
            xmax = tile.t_xmax,
            ymin = tile.t_ymin,
            ymax = tile.t_ymax;
        ms1_kernel.setArg(0,  xmin);
        ms1_kernel.setArg(1,  xmax);
        ms1_kernel.setArg(2,  ymin);
        ms1_kernel.setArg(3,  ymax);
        ms1_kernel.setArg(4,  *tile.field.d_work_array5);
        ms1_kernel.setArg(5,  *tile.field.d_work_array6);
        ms1_kernel.setArg(6,  *tile.field.d_volume);
        ms1_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms1_kernel.setArg(8,  *tile.field.d_vol_flux_y);

        openclQueue.enqueueNDRangeKernel(ms1_kernel, cl::NullRange, cl::NDRange((xmax + 2) - (xmin - 2) + 1, (ymax + 2) - (ymin - 2) + 1), cl::NullRange);
    } else if (mom_sweep == 2) {
        cl::Kernel ms2_kernel(openclProgram, "ms2_kernel");

        int xmin = tile.t_xmin,
            xmax = tile.t_xmax,
            ymin = tile.t_ymin,
            ymax = tile.t_ymax;
        ms2_kernel.setArg(0,  xmin);
        ms2_kernel.setArg(1,  xmax);
        ms2_kernel.setArg(2,  ymin);
        ms2_kernel.setArg(3,  ymax);
        ms2_kernel.setArg(4,  *tile.field.d_work_array5);
        ms2_kernel.setArg(5,  *tile.field.d_work_array6);
        ms2_kernel.setArg(6,  *tile.field.d_volume);
        ms2_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms2_kernel.setArg(8,  *tile.field.d_vol_flux_y);

        openclQueue.enqueueNDRangeKernel(ms2_kernel, cl::NullRange, cl::NDRange((xmax + 2) - (xmin - 2) + 1, (ymax + 2) - (ymin - 2) + 1), cl::NullRange);
    } else if (mom_sweep == 3) {
        cl::Kernel ms3_kernel(openclProgram, "ms3_kernel");

        int xmin = tile.t_xmin,
            xmax = tile.t_xmax,
            ymin = tile.t_ymin,
            ymax = tile.t_ymax;
        ms3_kernel.setArg(0,  xmin);
        ms3_kernel.setArg(1,  xmax);
        ms3_kernel.setArg(2,  ymin);
        ms3_kernel.setArg(3,  ymax);
        ms3_kernel.setArg(4,  *tile.field.d_work_array5);
        ms3_kernel.setArg(5,  *tile.field.d_work_array6);
        ms3_kernel.setArg(6,  *tile.field.d_volume);
        ms3_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms3_kernel.setArg(8,  *tile.field.d_vol_flux_y);

        openclQueue.enqueueNDRangeKernel(ms3_kernel, cl::NullRange, cl::NDRange((xmax + 2) - (xmin - 2) + 1, (ymax + 2) - (ymin - 2) + 1), cl::NullRange);
    } else if (mom_sweep == 4) {
        cl::Kernel ms4_kernel(openclProgram, "ms4_kernel");

        int xmin = tile.t_xmin,
            xmax = tile.t_xmax,
            ymin = tile.t_ymin,
            ymax = tile.t_ymax;
        ms4_kernel.setArg(0,  xmin);
        ms4_kernel.setArg(1,  xmax);
        ms4_kernel.setArg(2,  ymin);
        ms4_kernel.setArg(3,  ymax);
        ms4_kernel.setArg(4,  *tile.field.d_work_array5);
        ms4_kernel.setArg(5,  *tile.field.d_work_array6);
        ms4_kernel.setArg(6,  *tile.field.d_volume);
        ms4_kernel.setArg(7,  *tile.field.d_vol_flux_x);
        ms4_kernel.setArg(8,  *tile.field.d_vol_flux_y);

        openclQueue.enqueueNDRangeKernel(ms4_kernel, cl::NullRange, cl::NDRange((xmax + 2) - (xmin - 2) + 1, (ymax + 2) - (ymin - 2) + 1), cl::NullRange);
    }
    openclQueue.finish();

    if (direction == 1) {
        cl::Kernel dx1_kernel(openclProgram, "dx1_kernel");

        int xmin = tile.t_xmin,
            xmax = tile.t_xmax,
            ymin = tile.t_ymin,
            ymax = tile.t_ymax;
        dx1_kernel.setArg(0,  xmin);
        dx1_kernel.setArg(1,  xmax);
        dx1_kernel.setArg(2,  ymin);
        dx1_kernel.setArg(3,  ymax);
        dx1_kernel.setArg(4,  *tile.field.d_work_array1);
        dx1_kernel.setArg(5,  *tile.field.d_mass_flux_x);

        openclQueue.enqueueNDRangeKernel(dx1_kernel, cl::NullRange, cl::NDRange((xmax + 2) - (xmin - 2) + 1, (ymax + 1) - (ymin) + 1), cl::NullRange);

        cl::Kernel dx2_kernel(openclProgram, "dx2_kernel");

        dx2_kernel.setArg(0,  xmin);
        dx2_kernel.setArg(1,  xmax);
        dx2_kernel.setArg(2,  ymin);
        dx2_kernel.setArg(3,  ymax);
        dx2_kernel.setArg(4,  *tile.field.d_work_array2);
        dx2_kernel.setArg(5,  *tile.field.d_work_array3);
        dx2_kernel.setArg(6,  *tile.field.d_density1);
        dx2_kernel.setArg(7,  *tile.field.d_work_array6);
        dx2_kernel.setArg(8,  *tile.field.d_work_array1);

        openclQueue.enqueueNDRangeKernel(dx2_kernel, cl::NullRange, cl::NDRange((xmax + 2) - (xmin - 1) + 1, (ymax + 1) - (ymin) + 1), cl::NullRange);

        openclQueue.finish();


        // cl::Kernel dx3_kernel(openclProgram, "dx3_kernel");

        // dx3_kernel.setArg(0,  xmin);
        // dx3_kernel.setArg(1,  xmax);
        // dx3_kernel.setArg(2,  ymin);
        // dx3_kernel.setArg(3,  ymax);
        // dx3_kernel.setArg(4,  *tile.field.d_work_array4);
        // dx3_kernel.setArg(5,  *tile.field.d_work_array1);
        // dx3_kernel.setArg(6,  *tile.field.d_celldx);
        // dx3_kernel.setArg(7,  *tile.field.);
        // dx3_kernel.setArg(8,  *tile.field.d_work_array1);
        //     global double* mom_flux,
        // const global double* node_flux,
        // const global double* node_mass_pre,
        // const global double* celldx,
        // const global double* vel1

        // openclQueue.enqueueNDRangeKernel(dx3_kernel, cl::NullRange, cl::NDRange((xmax + 1) - (xmin - 1) + 1, (ymax + 1) - (ymin) + 1), cl::NullRange);

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
    } else if (direction == 2) {
        cl::Kernel dy1_kernel(openclProgram, "dy1_kernel");

        int xmin = tile.t_xmin,
            xmax = tile.t_xmax,
            ymin = tile.t_ymin,
            ymax = tile.t_ymax;
        dy1_kernel.setArg(0,  xmin);
        dy1_kernel.setArg(1,  xmax);
        dy1_kernel.setArg(2,  ymin);
        dy1_kernel.setArg(3,  ymax);
        dy1_kernel.setArg(4,  *tile.field.d_work_array1);
        dy1_kernel.setArg(5,  *tile.field.d_mass_flux_y);

        openclQueue.enqueueNDRangeKernel(dy1_kernel, cl::NullRange, cl::NDRange((xmax + 1) - (xmin) + 1, (ymax + 2) - (ymin - 2) + 1), cl::NullRange);

        cl::Kernel dy2_kernel(openclProgram, "dy2_kernel");

        dy2_kernel.setArg(0,  xmin);
        dy2_kernel.setArg(1,  xmax);
        dy2_kernel.setArg(2,  ymin);
        dy2_kernel.setArg(3,  ymax);
        dy2_kernel.setArg(4,  *tile.field.d_work_array2);
        dy2_kernel.setArg(5,  *tile.field.d_work_array3);
        dy2_kernel.setArg(6,  *tile.field.d_density1);
        dy2_kernel.setArg(7,  *tile.field.d_work_array6);
        dy2_kernel.setArg(8,  *tile.field.d_work_array1);

        openclQueue.enqueueNDRangeKernel(dy2_kernel, cl::NullRange, cl::NDRange((xmax + 1) - (xmin) + 1, (ymax + 2) - (ymin - 1) + 1), cl::NullRange);
        openclQueue.finish();


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
#endif