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
 *  @brief C flux kernel.
 *  @author Wayne Gaudin
 *  @details The edge volume fluxes are calculated based on the velocity fields.
 */

#include "ftocmacros.h"
#include "../definitions_c.h"

void flux_calc_x_kernel(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const double * __restrict__ xarea,
    const double * __restrict__ xvel0,
    const double * __restrict__ xvel1,
    double * __restrict__ vol_flux_x)
{
    vol_flux_x[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] =
        0.25 * dt * xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
        * (xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
           + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
           + xvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
           + xvel1[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]);
}

void flux_calc_y_kernel(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const double * __restrict__ yarea,
    const double * __restrict__ yvel0,
    const double * __restrict__ yvel1,
    double * __restrict__ vol_flux_y)
{
    vol_flux_y[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] =
        0.25 * dt * yarea[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
        * (yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
           + yvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
           + yvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
           + yvel1[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]);
}
