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

// #include <stdio.h>
// #include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif



void advec_cell_kernel_c_(
// int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    const double* __restrict__ vertexdx,
    const double* __restrict__ vertexdy,
    const double* __restrict__ volume,
    double* __restrict__ density1,
    double* __restrict__ energy1,
    double* __restrict__ mass_flux_x,
    const double* __restrict__ vol_flux_x,
    double* __restrict__ mass_flux_y,
    const double* __restrict__ vol_flux_y,
    double* __restrict__ pre_vol,
    double* __restrict__ post_vol,
    double* __restrict__ pre_mass,
    double* __restrict__ post_mass,
    double* __restrict__ advec_vol,
    double* __restrict__ post_ener,
    double* __restrict__ ener_flux,
    int dir,
    int sweep_number)
{
    int g_xdir = 1, g_ydir = 2;

    double one_by_six;

    one_by_six = 1.0 / 6.0;
    #pragma omp parallel
    {
        if (dir == g_xdir) {
            if (sweep_number == 1) {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                    + (VOL_FLUX_X(vol_flux_x, j + 1, k)
                    - VOL_FLUX_X(vol_flux_x, j, k)
                    + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                    - VOL_FLUX_Y(vol_flux_y, j, k));
                    WORK_ARRAY(post_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                    - (VOL_FLUX_X(vol_flux_x, j + 1, k)
                    - VOL_FLUX_X(vol_flux_x, j, k));
                });

            } else {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                    + VOL_FLUX_X(vol_flux_x, j + 1, k)
                    - VOL_FLUX_X(vol_flux_x, j, k);
                    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
                });

            }

            DOUBLEFOR(y_min, y_max, x_min, x_max + 2, ({
                int upwind, donor, downwind, dif;
                if (VOL_FLUX_X(vol_flux_x, j, k) > 0.0)
                {
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
                double sigma3 = (1.0 + sigmat) * (vertexdx[FTNREF1D(j, x_min - 2)] / vertexdx[FTNREF1D(dif, x_min - 2)]);
                double sigma4 = 2.0 - sigmat;

                double sigma = sigmat;
                double sigmav = sigmat;

                double diffuw = DENSITY1(density1, donor, k) - DENSITY1(density1, upwind, k);
                double diffdw = DENSITY1(density1, downwind, k) - DENSITY1(density1, donor, k);
                double limiter;
                if (diffuw * diffdw > 0.0)
                {
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
                if (diffuw * diffdw > 0.0)
                {
                    limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                    , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
                } else {
                    limiter = 0.0;
                }
                WORK_ARRAY(ener_flux, j, k) = MASS_FLUX_X(mass_flux_x, j, k)
                * (ENERGY1(energy1, donor, k) + limiter);

            }));


            DOUBLEFOR(y_min, y_max,
            x_min, x_max, {
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
            });

        } else if (dir == g_ydir) {
            if (sweep_number == 1) {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                    + (VOL_FLUX_Y(vol_flux_y, j, k + 1)
                    - VOL_FLUX_Y(vol_flux_y, j, k)
                    + VOL_FLUX_X(vol_flux_x, j + 1, k)
                    - VOL_FLUX_X(vol_flux_x, j, k));
                    WORK_ARRAY(post_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                    - (VOL_FLUX_Y(vol_flux_y, j, k + 1)
                    - VOL_FLUX_Y(vol_flux_y, j, k));
                });

            } else {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                    + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                    - VOL_FLUX_Y(vol_flux_y, j, k);
                    WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
                });

            }

            DOUBLEFOR(y_min, y_max + 2, x_min, x_max, ({
                int upwind, donor, downwind, dif;
                if (VOL_FLUX_Y(vol_flux_y, j, k) > 0.0)
                {
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
                double sigma3 = (1.0 + sigmat) * (vertexdy[FTNREF1D(k, y_min - 2)] / vertexdy[FTNREF1D(dif, y_min - 2)]);
                double sigma4 = 2.0 - sigmat;

                double sigma = sigmat;
                double sigmav = sigmat;

                double diffuw = DENSITY1(density1, j, donor) - DENSITY1(density1, j, upwind);
                double diffdw = DENSITY1(density1, j, downwind) - DENSITY1(density1, j, donor);
                double limiter;
                if (diffuw * diffdw > 0.0)
                {
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
                if (diffuw * diffdw > 0.0)
                {
                    limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                    , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
                } else {
                    limiter = 0.0;
                }
                WORK_ARRAY(ener_flux, j, k) = MASS_FLUX_Y(mass_flux_y, j, k)
                * (ENERGY1(energy1, j, donor) + limiter);

            }));

            DOUBLEFOR(y_min, y_max,
            x_min, x_max, {
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
            });
        }
    }

#ifdef USE_KOKKOS
    Kokkos::fence();
#endif
}
