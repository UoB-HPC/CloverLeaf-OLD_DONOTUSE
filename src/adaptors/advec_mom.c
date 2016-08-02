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
#endif