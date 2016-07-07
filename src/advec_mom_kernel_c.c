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
 *  @brief C momentum advection kernel
 *  @author Wayne Gaudin
 *  @details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "definitions_c.h"

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif


void advec_mom_kernel_c_(
    struct tile_type *tile,
    double *vel1,
    int sweep_number,
    int direction)
{
    int x_min = tile->t_xmin,
        x_max = tile->t_xmax,
        y_min = tile->t_ymin,
        y_max = tile->t_ymax;

    double *mass_flux_x = tile->field.mass_flux_x;
    double *vol_flux_x = tile->field.vol_flux_x;
    double *mass_flux_y = tile->field.mass_flux_y;
    double *vol_flux_y = tile->field.vol_flux_y;
    double *volume = tile->field.volume;
    double *density1 = tile->field.density1;
    double *node_flux = tile->field.work_array1;
    double *node_mass_post = tile->field.work_array2;
    double *node_mass_pre = tile->field.work_array3;
    double *mom_flux = tile->field.work_array4;
    double *pre_vol = tile->field.work_array5;
    double *post_vol = tile->field.work_array6;

    double *celldx = tile->field.celldx;
    double *celldy = tile->field.celldy;

    int k, mom_sweep;

    mom_sweep = direction + 2 * (sweep_number - 1);
    #pragma omp parallel
    {
        if (mom_sweep == 1) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {

                post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                + vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
                pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            });
        } else if (mom_sweep == 2) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {

                post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                + vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
                pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];

            });
        } else if (mom_sweep == 3) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {

                post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
                pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + vol_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                - vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];

            });
        } else if (mom_sweep == 4) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {

                post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
                pre_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + vol_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                - vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            });
        }

        if (direction == 1) {
            DOUBLEFOR(y_min, y_max + 1, x_min - 2, x_max + 2, {

                node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = 0.25
                * (mass_flux_x[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)]
                + mass_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + mass_flux_x[FTNREF2D(j + 1, k - 1, x_max + 5, x_min - 2, y_min - 2)]
                + mass_flux_x[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]);

            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 2, {

                node_mass_post[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = 0.25
                * (density1[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)]
                + density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + density1[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j - 1, k - 1, x_max + 5, x_min - 2, y_min - 2)]
                + density1[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j - 1, k  , x_max + 5, x_min - 2, y_min - 2)]);

            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 2, {

                node_mass_pre[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = node_mass_post[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                - node_flux[FTNREF2D(j - 1, k  , x_max + 5, x_min - 2, y_min - 2)] + node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            });

            DOUBLEFOR(y_min, y_max + 1, x_min - 1, x_max + 1, ({

                int upwind, donor, downwind, dif;
                if (node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] < 0.0)
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
                double sigma = fabs(node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]) / (node_mass_pre[FTNREF2D(donor, k  , x_max + 5, x_min - 2, y_min - 2)]);
                double width = celldx[FTNREF1D(j, x_min - 2)];
                double vdiffuw = vel1[FTNREF2D(donor, k  , x_max + 5, x_min - 2, y_min - 2)] - vel1[FTNREF2D(upwind, k  , x_max + 5, x_min - 2, y_min - 2)];
                double vdiffdw = vel1[FTNREF2D(downwind, k  , x_max + 5, x_min - 2, y_min - 2)] - vel1[FTNREF2D(donor, k  , x_max + 5, x_min - 2, y_min - 2)];
                double limiter = 0.0;
                if (vdiffuw * vdiffdw > 0.0)
                {
                    double auw = fabs(vdiffuw);
                    double adw = fabs(vdiffdw);
                    double wind = 1.0;
                    if (vdiffdw <= 0.0) wind = -1.0;
                    limiter = wind * MIN(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldx[FTNREF1D(dif, x_min - 2)]) / 6.0, MIN(auw, adw));
                }
                double advec_vel_s = vel1[FTNREF2D(donor, k  , x_max + 5, x_min - 2, y_min - 2)] + (1.0 - sigma) * limiter;
                mom_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = advec_vel_s
                * node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            }));

            DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, {

                vel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = (vel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                * node_mass_pre[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + mom_flux[FTNREF2D(j - 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                - mom_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)])
                / node_mass_post[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            });
        } else if (direction == 2) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min , x_max + 1, {

                node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = 0.25
                * (mass_flux_y[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)]
                + mass_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                + mass_flux_y[FTNREF2D(j - 1, k + 1, x_max + 4, x_min - 2, y_min - 2)]
                + mass_flux_y[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]);

            });

            DOUBLEFOR(y_min - 1, y_max + 2, x_min, x_max + 1, {

                node_mass_post[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = 0.25
                * (density1[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)]
                + density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + density1[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j - 1, k - 1, x_max + 5, x_min - 2, y_min - 2)]
                + density1[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)]
                * post_vol[FTNREF2D(j - 1, k  , x_max + 5, x_min - 2, y_min - 2)]);

            });

            DOUBLEFOR(y_min - 1, y_max + 2, x_min, x_max + 1, {

                node_mass_pre[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = node_mass_post[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                - node_flux[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)] + node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            });

            DOUBLEFOR(y_min - 1, y_max + 1, x_min , x_max + 1, ({

                int upwind, donor, downwind, dif;
                if (node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] < 0.0)
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
                double sigma = fabs(node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]) / (node_mass_pre[FTNREF2D(j  , donor, x_max + 5, x_min - 2, y_min - 2)]);
                double width = celldy[FTNREF1D(k, y_min - 2)];
                double vdiffuw = vel1[FTNREF2D(j  , donor, x_max + 5, x_min - 2, y_min - 2)] - vel1[FTNREF2D(j  , upwind, x_max + 5, x_min - 2, y_min - 2)];
                double vdiffdw = vel1[FTNREF2D(j  , downwind , x_max + 5, x_min - 2, y_min - 2)] - vel1[FTNREF2D(j  , donor, x_max + 5, x_min - 2, y_min - 2)];
                double limiter = 0.0;
                if (vdiffuw * vdiffdw > 0.0)
                {
                    double auw = fabs(vdiffuw);
                    double adw = fabs(vdiffdw);
                    double wind = 1.0;
                    if (vdiffdw <= 0.0) wind = -1.0;
                    limiter = wind * MIN(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldy[FTNREF1D(dif, y_min - 2)]) / 6.0, MIN(auw, adw));
                }
                double advec_vel_s = vel1[FTNREF2D(j  , donor, x_max + 5, x_min - 2, y_min - 2)] + (1.0 - sigma) * limiter;
                mom_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = advec_vel_s
                * node_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            }));

            DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, {

                vel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = (vel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                * node_mass_pre[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                + mom_flux[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)]
                - mom_flux[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)])
                / node_mass_post[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];

            });
        }
    }

#ifdef USE_KOKKOS
    Kokkos::fence();
#endif


}
