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
 *  @brief C PdV kernel.
 *  @author Wayne Gaudin
 *  @details Calculates the change in energy and density in a cell using the
 *  change on cell volume due to the velocity gradients in a cell. The time
 *  level of the velocity data depends on whether it is invoked as the
 *  predictor or corrector.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void pdv_kernel_predict_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    double dt,
    const double * __restrict__ xarea,
    const double * __restrict__ yarea,
    const double * __restrict__ volume,
    const double * __restrict__ density0,
    double * __restrict__ density1,
    const double * __restrict__ energy0,
    double * __restrict__ energy1,
    const double * __restrict__ pressure,
    const double * __restrict__ viscosity,
    const double * __restrict__ xvel0,
    const double * __restrict__ xvel1,
    const double * __restrict__ yvel0,
    const double * __restrict__ yvel1,
    double * __restrict__ volume_change)
{
    double left_flux =  (xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)])
                        * (xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)])
                        * 0.25 * dt * 0.5;
    double right_flux = (xarea[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)])
                        * (xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                        * 0.25 * dt * 0.5;
    double bottom_flux = (yarea[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)])
                         * (yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                            + yvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                            + yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                            + yvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)])
                         * 0.25 * dt * 0.5;
    double top_flux =   (yarea[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)])
                        * (yvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + yvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                        * 0.25 * dt * 0.5;

    double total_flux = right_flux - left_flux + top_flux - bottom_flux;

    volume_change[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            / (volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + total_flux);

    // min_cell_volume = MIN(volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + right_flux - left_flux + top_flux - bottom_flux
    //                       , MIN(volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + right_flux - left_flux
    //                             , volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + top_flux - bottom_flux));

    double recip_volume = 1.0 / volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];

    double energy_change = (pressure[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] / density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                            + viscosity[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] / density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)])
                           * total_flux * recip_volume;

    energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = energy0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] - energy_change;

    density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            * volume_change[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
}
void pdv_kernel_no_predict_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    double dt,
    const double * __restrict__ xarea,
    const double * __restrict__ yarea,
    const double * __restrict__ volume,
    const double * __restrict__ density0,
    double * __restrict__ density1,
    const double * __restrict__ energy0,
    double * __restrict__ energy1,
    const double * __restrict__ pressure,
    const double * __restrict__ viscosity,
    const double * __restrict__ xvel0,
    const double * __restrict__ xvel1,
    const double * __restrict__ yvel0,
    const double * __restrict__ yvel1,
    double * __restrict__ volume_change)
{

    double left_flux =  (xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)])
                        * (xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + xvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel1[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)])
                        * 0.25 * dt;
    double right_flux = (xarea[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)])
                        * (xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + xvel1[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                           + xvel1[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                        * 0.25 * dt;
    double bottom_flux = (yarea[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)])
                         * (yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                            + yvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                            + yvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                            + yvel1[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)])
                         * 0.25 * dt;
    double top_flux =   (yarea[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)])
                        * (yvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + yvel1[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                           + yvel1[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                        * 0.25 * dt;

    double total_flux = right_flux - left_flux + top_flux - bottom_flux;

    volume_change[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            / (volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + total_flux);

    // min_cell_volume = MIN(volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + right_flux - left_flux + top_flux - bottom_flux
    //                       , MIN(volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + right_flux - left_flux
    //                             , volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] + top_flux - bottom_flux));

    double recip_volume = 1.0 / volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];

    double energy_change = (pressure[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] / density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
                            + viscosity[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] / density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)])
                           * total_flux * recip_volume;

    energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = energy0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] - energy_change;

    density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            * volume_change[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
}

