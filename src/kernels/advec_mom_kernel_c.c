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
    double* __restrict__ pre_vol,
    double* __restrict__ post_vol ,
    const double* __restrict__ volume ,
    const double* __restrict__ vol_flux_x ,
    const double* __restrict__ vol_flux_y)
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
    double* __restrict__ pre_vol,
    double* __restrict__ post_vol ,
    const double* __restrict__ volume ,
    const double* __restrict__ vol_flux_x ,
    const double* __restrict__ vol_flux_y)
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
    double* __restrict__ pre_vol,
    double* __restrict__ post_vol ,
    const double* __restrict__ volume ,
    const double* __restrict__ vol_flux_x ,
    const double* __restrict__ vol_flux_y)
{
    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    WORK_ARRAY(pre_vol, j, k) = WORK_ARRAY(post_vol, j, k)
                                + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                - VOL_FLUX_Y(vol_flux_y, j, k);
}

void ms4(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    double* __restrict__ pre_vol,
    double* __restrict__ post_vol ,
    const double* __restrict__ volume ,
    const double* __restrict__ vol_flux_x ,
    const double* __restrict__ vol_flux_y)
{
    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    WORK_ARRAY(pre_vol, j, k) = WORK_ARRAY(post_vol, j, k)
                                + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                - VOL_FLUX_X(vol_flux_x, j, k);
}

void advec_mom_kernel_c_(
    double* vel1,
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    const double* __restrict__ mass_flux_x,
    const double* __restrict__ vol_flux_x ,
    const double* __restrict__ mass_flux_y,
    const double* __restrict__ vol_flux_y ,
    const double* __restrict__ volume ,
    const double* __restrict__ density1 ,
    double* __restrict__ node_flux,
    double* __restrict__ node_mass_post ,
    double* __restrict__ node_mass_pre,
    double* __restrict__ mom_flux ,
    double* __restrict__ pre_vol,
    double* __restrict__ post_vol ,
    const double* __restrict__ celldx,
    const double* __restrict__ celldy,
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

                WORK_ARRAY(node_flux, j, k) = 0.25
                * (MASS_FLUX_X(mass_flux_x, j, k - 1)
                + MASS_FLUX_X(mass_flux_x, j, k)
                + MASS_FLUX_X(mass_flux_x, j + 1, k - 1)
                + MASS_FLUX_X(mass_flux_x, j + 1, k));

            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 2, {

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

            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 1, ({

                int upwind, donor, downwind, dif;
                if (WORK_ARRAY(node_flux, j, k) < 0.0)
                {
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
                double width = celldx[FTNREF1D(j, x_min - 2)];
                double vdiffuw = VEL(vel1, donor, k) - VEL(vel1, upwind, k);
                double vdiffdw = VEL(vel1, downwind, k) - VEL(vel1, donor, k);
                double limiter = 0.0;
                if (vdiffuw * vdiffdw > 0.0)
                {
                    double auw = fabs(vdiffuw);
                    double adw = fabs(vdiffdw);
                    double wind = 1.0;
                    if (vdiffdw <= 0.0) wind = -1.0;
                    limiter = wind * MIN(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldx[FTNREF1D(dif, x_min - 2)]) / 6.0, MIN(auw, adw));
                }
                double advec_vel_s = VEL(vel1, donor, k) + (1.0 - sigma) * limiter;
                WORK_ARRAY(mom_flux, j, k) = advec_vel_s
                * WORK_ARRAY(node_flux, j, k);

            }));

            DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, {

                VEL(vel1, j, k) = (VEL(vel1, j, k)
                * WORK_ARRAY(node_mass_pre, j, k)
                + WORK_ARRAY(mom_flux, j - 1, k)
                - WORK_ARRAY(mom_flux, j, k))
                / WORK_ARRAY(node_mass_post, j, k);

            });
        } else if (direction == 2) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min , x_max + 1, {

                WORK_ARRAY(node_flux, j, k) = 0.25
                * (MASS_FLUX_Y(mass_flux_y, j - 1, k)
                + MASS_FLUX_Y(mass_flux_y, j, k)
                + MASS_FLUX_Y(mass_flux_y, j - 1, k + 1)
                + MASS_FLUX_Y(mass_flux_y, j, k + 1));

            });

            DOUBLEFOR(y_min - 1, y_max + 2, x_min, x_max + 1, {

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

            });

            DOUBLEFOR(y_min - 1, y_max + 1, x_min , x_max + 1, ({

                int upwind, donor, downwind, dif;
                if (WORK_ARRAY(node_flux, j, k) < 0.0)
                {
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
                double width = celldy[FTNREF1D(k, y_min - 2)];
                double vdiffuw = VEL(vel1, j, donor) - VEL(vel1, j, upwind);
                double vdiffdw = VEL(vel1, j, downwind) - VEL(vel1, j, donor);
                double limiter = 0.0;
                if (vdiffuw * vdiffdw > 0.0)
                {
                    double auw = fabs(vdiffuw);
                    double adw = fabs(vdiffdw);
                    double wind = 1.0;
                    if (vdiffdw <= 0.0) wind = -1.0;
                    limiter = wind * MIN(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldy[FTNREF1D(dif, y_min - 2)]) / 6.0, MIN(auw, adw));
                }
                double advec_vel_s = VEL(vel1, j, donor) + (1.0 - sigma) * limiter;
                WORK_ARRAY(mom_flux, j, k) = advec_vel_s
                * WORK_ARRAY(node_flux, j, k);

            }));

            DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, {

                VEL(vel1, j, k) = (VEL(vel1, j, k)
                * WORK_ARRAY(node_mass_pre, j, k)
                + WORK_ARRAY(mom_flux, j, k - 1)
                - WORK_ARRAY(mom_flux, j, k))
                / WORK_ARRAY(node_mass_post, j, k);

            });
        }
    }

#ifdef USE_KOKKOS
    Kokkos::fence();
#endif


}
