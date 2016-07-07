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
 *  @brief C cell advection kernel.
 *  @author Wayne Gaudin
 *  @details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "definitions_c.h"

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif



void advec_cell_kernel_c_(
    struct tile_type *tile,
    int dir,
    int sweep_number)
{
    int x_min = tile->t_xmin,
        x_max = tile->t_xmax,
        y_min = tile->t_ymin,
        y_max = tile->t_ymax;

    double *vertexdx = tile->field.vertexdx;
    double *vertexdy = tile->field.vertexdy;
    double *volume = tile->field.volume;
    double *density1 = tile->field.density1;
    double *energy1 = tile->field.energy1;
    double *mass_flux_x = tile->field.mass_flux_x;
    double *vol_flux_x = tile->field.vol_flux_x;
    double *mass_flux_y = tile->field.mass_flux_y;
    double *vol_flux_y = tile->field.vol_flux_y;
    double *pre_vol = tile->field.work_array1;
    double *post_vol = tile->field.work_array2;
    double *pre_mass = tile->field.work_array3;
    double *post_mass = tile->field.work_array4;
    double *advec_vol = tile->field.work_array5;
    double *post_ener = tile->field.work_array6;
    double *ener_flux = tile->field.work_array7;

    int j, k, upwind, donor, downwind, dif;

    int g_xdir = 1, g_ydir = 2;

    double one_by_six;

    one_by_six = 1.0 / 6.0;
    #pragma omp parallel
    {
        if (dir == g_xdir) {
            if (sweep_number == 1) {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                    + (vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                    - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                    + vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                    - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]);
                    post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                    - (vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                    - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]);
                });

            } else {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                    + vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                    - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                    post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
                });

            }

            DOUBLEFOR(y_min, y_max, x_min, x_max, ({
                int upwind, donor, downwind, dif;
                if (vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] > 0.0)
                {
                    upwind   = j - 2;
                    donor    = j - 1;
                    downwind = j;
                    dif      = donor;
                } else {
                    upwind   = MIN(j + 1, x_max + 2);
                    donor    = j;
                    downwind = j - 1;
                    dif      = upwind;
                }

                double sigmat = fabs(vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] / pre_vol[FTNREF2D(donor, k  , x_max + 5, x_min - 2, y_min - 2)]);
                double sigma3 = (1.0 + sigmat) * (vertexdx[FTNREF1D(j, x_min - 2)] / vertexdx[FTNREF1D(dif, x_min - 2)]);
                double sigma4 = 2.0 - sigmat;

                double sigma = sigmat;
                double sigmav = sigmat;

                double diffuw = density1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)] - density1[FTNREF2D(upwind, k  , x_max + 4, x_min - 2, y_min - 2)];
                double diffdw = density1[FTNREF2D(downwind, k  , x_max + 4, x_min - 2, y_min - 2)] - density1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)];
                double limiter;
                if (diffuw * diffdw > 0.0)
                {
                    limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                    , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
                } else {
                    limiter = 0.0;
                }
                mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] = vol_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)]
                * (density1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)] + limiter);

                double sigmam = fabs(mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)]) / (density1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)]
                * pre_vol[FTNREF2D(donor, k  , x_max + 5, x_min - 2, y_min - 2)]);
                diffuw = energy1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)] - energy1[FTNREF2D(upwind, k  , x_max + 4, x_min - 2, y_min - 2)];
                diffdw = energy1[FTNREF2D(downwind, k  , x_max + 4, x_min - 2, y_min - 2)] - energy1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)];
                if (diffuw * diffdw > 0.0)
                {
                    limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                    , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
                } else {
                    limiter = 0.0;
                }
                ener_flux[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] = mass_flux_x[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)]
                * (energy1[FTNREF2D(donor, k  , x_max + 4, x_min - 2, y_min - 2)] + limiter);

            }));


            DOUBLEFOR(y_min, y_max,
            x_min, x_max, {
                pre_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                * pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                post_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = pre_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + mass_flux_x[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]
                - mass_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)];
                post_ener[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = (energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                * pre_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + ener_flux[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]
                - ener_flux[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)])
                / post_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                advec_vol [FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + vol_flux_x[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]
                - vol_flux_x[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)];

                density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = post_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] / advec_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = post_ener[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
            });

        } else if (dir == g_ydir) {
            if (sweep_number == 1) {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                    + (vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                    - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                    + vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                    - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]);
                    post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                    - (vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                    - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]);
                });

            } else {

                DOUBLEFOR(y_min - 2, y_max + 2,
                x_min - 2, x_max + 2, {
                    pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                    + vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                    - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
                    post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
                });

            }

            DOUBLEFOR(y_min, y_max + 2, x_min, x_max, ({
                int upwind, donor, downwind, dif;
                if (vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] > 0.0)
                {
                    upwind   = k - 2;
                    donor    = k - 1;
                    downwind = k;
                    dif      = donor;
                } else {
                    upwind   = MIN(k + 1, y_max + 2);
                    donor    = k;
                    downwind = k - 1;
                    dif      = upwind;
                }

                double sigmat = fabs(vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / pre_vol[FTNREF2D(j  , donor, x_max + 5, x_min - 2, y_min - 2)]);
                double sigma3 = (1.0 + sigmat) * (vertexdy[FTNREF1D(k, y_min - 2)] / vertexdy[FTNREF1D(dif, y_min - 2)]);
                double sigma4 = 2.0 - sigmat;

                double sigma = sigmat;
                double sigmav = sigmat;

                double diffuw = density1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)] - density1[FTNREF2D(j  , upwind, x_max + 4, x_min - 2, y_min - 2)];
                double diffdw = density1[FTNREF2D(j  , downwind, x_max + 4, x_min - 2, y_min - 2)] - density1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)];
                double limiter;
                if (diffuw * diffdw > 0.0)
                {
                    limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                    , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
                } else {
                    limiter = 0.0;
                }
                mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] = vol_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]
                * (density1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)] + limiter);

                double sigmam = fabs(mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]) / (density1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)]
                * pre_vol[FTNREF2D(j  , donor, x_max + 5, x_min - 2, y_min - 2)]);
                diffuw = energy1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)] - energy1[FTNREF2D(j  , upwind, x_max + 4, x_min - 2, y_min - 2)];
                diffdw = energy1[FTNREF2D(j  , downwind, x_max + 4, x_min - 2, y_min - 2)] - energy1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)];
                if (diffuw * diffdw > 0.0)
                {
                    limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                    , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
                } else {
                    limiter = 0.0;
                }
                ener_flux[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] = mass_flux_y[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]
                * (energy1[FTNREF2D(j  , donor, x_max + 4, x_min - 2, y_min - 2)] + limiter);

            }));

            DOUBLEFOR(y_min, y_max,
            x_min, x_max, {
                pre_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                * pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                post_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = pre_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + mass_flux_y[FTNREF2D(j, k  , x_max + 4, x_min - 2, y_min - 2)]
                - mass_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)];
                post_ener[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = (energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                * pre_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + ener_flux[FTNREF2D(j, k  , x_max + 5, x_min - 2, y_min - 2)]
                - ener_flux[FTNREF2D(j, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                / post_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                advec_vol [FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + vol_flux_y[FTNREF2D(j, k  , x_max + 4, x_min - 2, y_min - 2)]
                - vol_flux_y[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)];

                density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = post_mass[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] / advec_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = post_ener[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
            });
        }
    }

#ifdef USE_KOKKOS
    Kokkos::fence();
#endif
}
