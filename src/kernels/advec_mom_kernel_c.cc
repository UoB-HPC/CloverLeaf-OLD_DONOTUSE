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

// #include "ftocmacros.h"
// #include <math.h>
// #include "../definitions_c.h"


kernelqual void ms1(
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

kernelqual void ms2(
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

kernelqual void ms3(
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

kernelqual void ms4(
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

kernelqual void dx1(
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

kernelqual void dy1(
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

kernelqual void dx2(
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

kernelqual void dy2(
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

kernelqual void dx3(
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

kernelqual void dy3(
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

kernelqual void dx4(
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

kernelqual void dy4(
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
