/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *@brief C momentum advection kernel
 *@author Wayne Gaudin
 *@details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
 */

#include "ftocmacros.h"
// #include <math.h>
#include "../definitions_c.h"

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

void ms1(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k)
                                 + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                 - VOL_FLUX_Y(vol_flux_y, j, k);
    WORK_ARRAY(pre_vol, j, k) = WORK_ARRAY(post_vol, j, k)
                                + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                - VOL_FLUX_X(vol_flux_x, j, k);
}

void ms2(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k)
                                 + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                 - VOL_FLUX_X(vol_flux_x, j, k);
    WORK_ARRAY(pre_vol, j, k) = WORK_ARRAY(post_vol, j, k)
                                + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                - VOL_FLUX_Y(vol_flux_y, j, k);
}

void ms3(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    WORK_ARRAY(pre_vol, j, k) = WORK_ARRAY(post_vol, j, k)
                                + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                - VOL_FLUX_Y(vol_flux_y, j, k);
}

void ms4(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    WORK_ARRAY(pre_vol, j, k) = WORK_ARRAY(post_vol, j, k)
                                + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                - VOL_FLUX_X(vol_flux_x, j, k);
}

void dx1(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t node_flux,
    const_field_2d_t mass_flux_x)
{
    WORK_ARRAY(node_flux, j, k) = 0.25
                                  * (MASS_FLUX_X(mass_flux_x, j, k - 1)
                                     + MASS_FLUX_X(mass_flux_x, j, k)
                                     + MASS_FLUX_X(mass_flux_x, j + 1, k - 1)
                                     + MASS_FLUX_X(mass_flux_x, j + 1, k));
}
void dy1(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t node_flux,
    const_field_2d_t mass_flux_y)
{
    WORK_ARRAY(node_flux, j, k) = 0.25
                                  * (MASS_FLUX_Y(mass_flux_y, j - 1, k)
                                     + MASS_FLUX_Y(mass_flux_y, j, k)
                                     + MASS_FLUX_Y(mass_flux_y, j - 1, k + 1)
                                     + MASS_FLUX_Y(mass_flux_y, j, k + 1));

}
void dx2(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t node_mass_post,
    field_2d_t node_mass_pre,
    const_field_2d_t density1,
    const_field_2d_t post_vol,
    const_field_2d_t node_flux)
{
    WORK_ARRAY(node_mass_post, j, k) = 0.25
                                       * (DENSITY1(density1, j, k - 1)
                                          * WORK_ARRAY(post_vol, j, k - 1)
                                          + DENSITY1(density1, j, k)
                                          * WORK_ARRAY(post_vol, j, k)
                                          + DENSITY1(density1, j - 1, k - 1)
                                          * WORK_ARRAY(post_vol, j - 1, k - 1)
                                          + DENSITY1(density1, j - 1, k)
                                          * WORK_ARRAY(post_vol, j - 1, k));

    WORK_ARRAY(node_mass_pre, j, k) = WORK_ARRAY(node_mass_post, j, k)
                                      - WORK_ARRAY(node_flux, j - 1, k) + WORK_ARRAY(node_flux, j, k);
}
void dy2(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t node_mass_post,
    field_2d_t node_mass_pre,
    const_field_2d_t density1,
    const_field_2d_t post_vol,
    const_field_2d_t node_flux)
{
    WORK_ARRAY(node_mass_post, j, k) = 0.25
                                       * (DENSITY1(density1, j, k - 1)
                                          * WORK_ARRAY(post_vol, j, k - 1)
                                          + DENSITY1(density1, j, k)
                                          * WORK_ARRAY(post_vol, j, k)
                                          + DENSITY1(density1, j - 1, k - 1)
                                          * WORK_ARRAY(post_vol, j - 1, k - 1)
                                          + DENSITY1(density1, j - 1, k)
                                          * WORK_ARRAY(post_vol, j - 1, k));

    WORK_ARRAY(node_mass_pre, j, k) = WORK_ARRAY(node_mass_post, j, k)
                                      - WORK_ARRAY(node_flux, j, k - 1) + WORK_ARRAY(node_flux, j, k);

}

void dx3(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t mom_flux,
    const_field_2d_t node_flux,
    const_field_2d_t node_mass_pre,
    const_field_1d_t celldx,
    const_field_2d_t vel1)
{
    int upwind, donor, downwind, dif;
    if (WORK_ARRAY(node_flux, j, k) < 0.0) {
        upwind = j + 2;
        donor = j + 1;
        downwind = j;
        dif = donor;
    } else {
        upwind = j - 1;
        donor = j;
        downwind = j + 1;
        dif = upwind;
    }
    double sigma = fabs(WORK_ARRAY(node_flux, j, k)) / (WORK_ARRAY(node_mass_pre, donor, k));
    double width = FIELD_1D(celldx, j,  x_min - 2);
    double vdiffuw = VEL(vel1, donor, k) - VEL(vel1, upwind, k);
    double vdiffdw = VEL(vel1, downwind, k) - VEL(vel1, donor, k);
    double limiter = 0.0;
    if (vdiffuw * vdiffdw > 0.0) {
        double auw = fabs(vdiffuw);
        double adw = fabs(vdiffdw);
        double wind = 1.0;
        if (vdiffdw <= 0.0) wind = -1.0;
        limiter = wind * MIN(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / FIELD_1D(celldx, dif,  x_min - 2)) / 6.0, MIN(auw, adw));
    }
    double advec_vel_s = VEL(vel1, donor, k) + (1.0 - sigma) * limiter;
    WORK_ARRAY(mom_flux, j, k) = advec_vel_s
                                 * WORK_ARRAY(node_flux, j, k);
}
void dy3(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t mom_flux,
    const_field_2d_t node_flux,
    const_field_2d_t node_mass_pre,
    const_field_1d_t celldy,
    const_field_2d_t vel1)
{
    int upwind, donor, downwind, dif;
    if (WORK_ARRAY(node_flux, j, k) < 0.0) {
        upwind = k + 2;
        donor = k + 1;
        downwind = k;
        dif = donor;
    } else {
        upwind = k - 1;
        donor = k;
        downwind = k + 1;
        dif = upwind;
    }
    double sigma = fabs(WORK_ARRAY(node_flux, j, k)) / (WORK_ARRAY(node_mass_pre, j, donor));
    double width = FIELD_1D(celldy, k,  y_min - 2);
    double vdiffuw = VEL(vel1, j, donor) - VEL(vel1, j, upwind);
    double vdiffdw = VEL(vel1, j, downwind) - VEL(vel1, j, donor);
    double limiter = 0.0;
    if (vdiffuw * vdiffdw > 0.0) {
        double auw = fabs(vdiffuw);
        double adw = fabs(vdiffdw);
        double wind = 1.0;
        if (vdiffdw <= 0.0) wind = -1.0;
        limiter = wind * MIN(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / FIELD_1D(celldy, dif,  y_min - 2)) / 6.0, MIN(auw, adw));
    }
    double advec_vel_s = VEL(vel1, j, donor) + (1.0 - sigma) * limiter;
    WORK_ARRAY(mom_flux, j, k) = advec_vel_s
                                 * WORK_ARRAY(node_flux, j, k);

}

void dx4(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t node_mass_pre,
    const_field_2d_t mom_flux,
    const_field_2d_t node_mass_post)
{
    VEL(vel1, j, k) = (VEL(vel1, j, k)
                       * WORK_ARRAY(node_mass_pre, j, k)
                       + WORK_ARRAY(mom_flux, j - 1, k)
                       - WORK_ARRAY(mom_flux, j, k))
                      / WORK_ARRAY(node_mass_post, j, k);
}
void dy4(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t node_mass_pre,
    const_field_2d_t mom_flux,
    const_field_2d_t node_mass_post)
{
    VEL(vel1, j, k) = (VEL(vel1, j, k)
                       * WORK_ARRAY(node_mass_pre, j, k)
                       + WORK_ARRAY(mom_flux, j, k - 1)
                       - WORK_ARRAY(mom_flux, j, k))
                      / WORK_ARRAY(node_mass_post, j, k);
}

void advec_mom_openmp(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t mass_flux_x,
    const_field_2d_t vol_flux_x,
    const_field_2d_t mass_flux_y,
    const_field_2d_t vol_flux_y,
    const_field_2d_t volume,
    const_field_2d_t density1,
    field_2d_t       node_flux,
    field_2d_t       node_mass_post,
    field_2d_t       node_mass_pre,
    field_2d_t       mom_flux,
    field_2d_t       pre_vol,
    field_2d_t       post_vol,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    int sweep_number,
    int direction);
void advec_mom_kokkos(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t mass_flux_x,
    const_field_2d_t vol_flux_x,
    const_field_2d_t mass_flux_y,
    const_field_2d_t vol_flux_y,
    const_field_2d_t volume,
    const_field_2d_t density1,
    field_2d_t       node_flux,
    field_2d_t       node_mass_post,
    field_2d_t       node_mass_pre,
    field_2d_t       mom_flux,
    field_2d_t       pre_vol,
    field_2d_t       post_vol,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    int sweep_number,
    int direction);

void advec_mom_kernel_c_(
    field_2d_t vel1,
    struct tile_type tile,
    int x_min,
    int x_max,
    int y_min,
    int y_max,
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

#if defined(USE_KOKKOS)
    advec_mom_kokkos(
        x_min, x_max, y_min, y_max,
        vel1,
        mass_flux_x,
        vol_flux_x,
        mass_flux_y,
        vol_flux_y,
        volume,
        density1,
        node_flux,
        node_mass_post,
        node_mass_pre,
        mom_flux,
        pre_vol,
        post_vol,
        celldx,
        celldy,
        sweep_number,
        direction);
#else
    advec_mom_openmp(
        x_min, x_max, y_min, y_max,
        vel1,
        mass_flux_x,
        vol_flux_x,
        mass_flux_y,
        vol_flux_y,
        volume,
        density1,
        node_flux,
        node_mass_post,
        node_mass_pre,
        mom_flux,
        pre_vol,
        post_vol,
        celldx,
        celldy,
        sweep_number,
        direction);
#endif

}

void advec_mom_openmp(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t mass_flux_x,
    const_field_2d_t vol_flux_x,
    const_field_2d_t mass_flux_y,
    const_field_2d_t vol_flux_y,
    const_field_2d_t volume,
    const_field_2d_t density1,
    field_2d_t       node_flux,
    field_2d_t       node_mass_post,
    field_2d_t       node_mass_pre,
    field_2d_t       mom_flux,
    field_2d_t       pre_vol,
    field_2d_t       post_vol,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    int sweep_number,
    int direction)
{
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

void advec_mom_kokkos(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t mass_flux_x,
    const_field_2d_t vol_flux_x,
    const_field_2d_t mass_flux_y,
    const_field_2d_t vol_flux_y,
    const_field_2d_t volume,
    const_field_2d_t density1,
    field_2d_t       node_flux,
    field_2d_t       node_mass_post,
    field_2d_t       node_mass_pre,
    field_2d_t       mom_flux,
    field_2d_t       pre_vol,
    field_2d_t       post_vol,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    int sweep_number,
    int direction)
{
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