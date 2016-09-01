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
 *@brief C cell advection kernel.
 *@author Wayne Gaudin
 *@details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
 */


// #include "ftocmacros.h"
// #include <math.h>
// #include "../definitions_c.h"


kernelqual void xsweep(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_vol,
    field_2d_t post_vol,
    const_field_2d_t volume,
    const_field_2d_t vol_flux_x,
    const_field_2d_t vol_flux_y,
    int sweep_number)
{
    if (sweep_number == 1) {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + (VOL_FLUX_X(vol_flux_x, j + 1, k)
                                       - VOL_FLUX_X(vol_flux_x, j, k)
                                       + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                       - VOL_FLUX_Y(vol_flux_y, j, k));
        WORK_ARRAY(post_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                     - (VOL_FLUX_X(vol_flux_x, j + 1, k)
                                        - VOL_FLUX_X(vol_flux_x, j, k));
    } else {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                    - VOL_FLUX_X(vol_flux_x, j, k);
        WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    }
}


kernelqual void ysweep(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_vol,
    field_2d_t post_vol,
    const_field_2d_t volume,
    const_field_2d_t vol_flux_x,
    const_field_2d_t vol_flux_y,
    int sweep_number)
{
    if (sweep_number == 1) {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + (VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                       - VOL_FLUX_Y(vol_flux_y, j, k)
                                       + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                       - VOL_FLUX_X(vol_flux_x, j, k));
        WORK_ARRAY(post_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                     - (VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                        - VOL_FLUX_Y(vol_flux_y, j, k));
    } else {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                    - VOL_FLUX_Y(vol_flux_y, j, k);
        WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    }
}

kernelqual void xcomp1(
    int j, int k,
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
    double one_by_six = 1.0 / 6.0;
    int upwind, donor, downwind, dif;
    if (VOL_FLUX_X(vol_flux_x, j, k) > 0.0) {
        upwind = j - 2;
        donor = j - 1;
        downwind = j;
        dif = donor;
    } else {
        upwind = MIN(j + 1, x_max + 2);
        donor = j;
        downwind = j - 1;
        dif = upwind;
    }

    double sigmat = fabs(VOL_FLUX_X(vol_flux_x, j, k) / WORK_ARRAY(pre_vol, donor, k));
    double sigma3 = (1.0 + sigmat) * (FIELD_1D(vertexdx, j,  x_min - 2) / FIELD_1D(vertexdx, dif,  x_min - 2));
    double sigma4 = 2.0 - sigmat;

    double sigmav = sigmat;

    double diffuw = DENSITY1(density1, donor, k) - DENSITY1(density1, upwind, k);
    double diffdw = DENSITY1(density1, downwind, k) - DENSITY1(density1, donor, k);
    double limiter;
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw),
                  one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    MASS_FLUX_X(mass_flux_x, j, k) = VOL_FLUX_X(vol_flux_x, j, k)
                                     * (DENSITY1(density1, donor, k) + limiter);

    double sigmam = fabs(MASS_FLUX_X(mass_flux_x, j, k)) / (DENSITY1(density1, donor, k)
                    * WORK_ARRAY(pre_vol, donor, k));
    diffuw = ENERGY1(energy1, donor, k) - ENERGY1(energy1, upwind, k);
    diffdw = ENERGY1(energy1, downwind, k) - ENERGY1(energy1, donor, k);
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                  , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    WORK_ARRAY(ener_flux, j, k) = MASS_FLUX_X(mass_flux_x, j, k)
                                  * (ENERGY1(energy1, donor, k) + limiter);
}


kernelqual void ycomp1(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mass_flux_y,
    field_2d_t ener_flux,
    const_field_2d_t vol_flux_y,
    const_field_2d_t pre_vol,
    const_field_2d_t density1,
    const_field_2d_t energy1,
    const_field_1d_t vertexdy)
{
    double one_by_six = 1.0 / 6.0;
    int upwind, donor, downwind, dif;
    if (VOL_FLUX_Y(vol_flux_y, j, k) > 0.0) {
        upwind = k - 2;
        donor = k - 1;
        downwind = k;
        dif = donor;
    } else {
        upwind = MIN(k + 1, y_max + 2);
        donor = k;
        downwind = k - 1;
        dif = upwind;
    }

    double sigmat = fabs(VOL_FLUX_Y(vol_flux_y, j, k) / WORK_ARRAY(pre_vol, j, donor));
    double sigma3 = (1.0 + sigmat) * (FIELD_1D(vertexdy, k,  y_min - 2) / FIELD_1D(vertexdy, dif,  y_min - 2));
    double sigma4 = 2.0 - sigmat;

    double sigmav = sigmat;

    double diffuw = DENSITY1(density1, j, donor) - DENSITY1(density1, j, upwind);
    double diffdw = DENSITY1(density1, j, downwind) - DENSITY1(density1, j, donor);
    double limiter;
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                  , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    MASS_FLUX_Y(mass_flux_y, j, k) = VOL_FLUX_Y(vol_flux_y, j, k)
                                     * (DENSITY1(density1, j, donor) + limiter);

    double sigmam = fabs(MASS_FLUX_Y(mass_flux_y, j, k)) / (DENSITY1(density1, j, donor)
                    * WORK_ARRAY(pre_vol, j, donor));
    diffuw = ENERGY1(energy1, j, donor) - ENERGY1(energy1, j, upwind);
    diffdw = ENERGY1(energy1, j, downwind) - ENERGY1(energy1, j, donor);
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                  , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    WORK_ARRAY(ener_flux, j, k) = MASS_FLUX_Y(mass_flux_y, j, k)
                                  * (ENERGY1(energy1, j, donor) + limiter);
}


kernelqual void xcomp2(
    int j, int k,
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
    WORK_ARRAY(pre_mass, j, k) = DENSITY1(density1, j, k)
                                 * WORK_ARRAY(pre_vol, j, k);
    WORK_ARRAY(post_mass, j, k) = WORK_ARRAY(pre_mass, j, k)
                                  + MASS_FLUX_X(mass_flux_x, j, k)
                                  - MASS_FLUX_X(mass_flux_x, j + 1, k);
    WORK_ARRAY(post_ener, j, k) = (ENERGY1(energy1, j, k)
                                   * WORK_ARRAY(pre_mass, j, k)
                                   + WORK_ARRAY(ener_flux, j, k)
                                   - WORK_ARRAY(ener_flux, j + 1, k))
                                  / WORK_ARRAY(post_mass, j, k);
    WORK_ARRAY(advec_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                  + VOL_FLUX_X(vol_flux_x, j, k)
                                  - VOL_FLUX_X(vol_flux_x, j + 1, k);

    DENSITY1(density1, j, k) = WORK_ARRAY(post_mass, j, k) / WORK_ARRAY(advec_vol, j, k);
    ENERGY1(energy1, j, k) = WORK_ARRAY(post_ener, j, k);
}

kernelqual void ycomp2(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_mass,
    field_2d_t post_mass,
    field_2d_t post_ener,
    field_2d_t advec_vol,
    field_2d_t density1,
    field_2d_t energy1,
    const_field_2d_t pre_vol,
    const_field_2d_t mass_flux_y,
    const_field_2d_t ener_flux,
    const_field_2d_t vol_flux_y)
{
    WORK_ARRAY(pre_mass, j, k) = DENSITY1(density1, j, k)
                                 * WORK_ARRAY(pre_vol, j, k);
    WORK_ARRAY(post_mass, j, k) = WORK_ARRAY(pre_mass, j, k)
                                  + MASS_FLUX_Y(mass_flux_y, j, k)
                                  - MASS_FLUX_Y(mass_flux_y, j, k + 1);
    WORK_ARRAY(post_ener, j, k) = (ENERGY1(energy1, j, k)
                                   * WORK_ARRAY(pre_mass, j, k)
                                   + WORK_ARRAY(ener_flux, j, k)
                                   - WORK_ARRAY(ener_flux, j, k + 1))
                                  / WORK_ARRAY(post_mass, j, k);
    WORK_ARRAY(advec_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                  + VOL_FLUX_Y(vol_flux_y, j, k)
                                  - VOL_FLUX_Y(vol_flux_y, j, k + 1);

    DENSITY1(density1, j, k) = WORK_ARRAY(post_mass, j, k) / WORK_ARRAY(advec_vol, j, k);
    ENERGY1(energy1, j, k) = WORK_ARRAY(post_ener, j, k);
}
