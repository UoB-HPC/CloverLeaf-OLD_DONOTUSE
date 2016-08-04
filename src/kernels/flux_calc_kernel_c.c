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
 *@brief C flux kernel.
 *@author Wayne Gaudin
 *@details The edge volume fluxes are calculated based on the velocity fields.
 */

// #include "ftocmacros.h"
// #include "../definitions_c.h"

void flux_calc_x_kernel(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    field_2d_t vol_flux_x)
{
    VOL_FLUX_X(vol_flux_x, j, k) =
        0.25 * dt * XAREA(xarea, j, k)
        * (XVEL0(xvel0, j, k)
           + XVEL0(xvel0, j, k + 1)
           + XVEL1(xvel1, j, k)
           + XVEL1(xvel1, j, k + 1));
}

void flux_calc_y_kernel(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t yarea,
    const_field_2d_t yvel0,
    const_field_2d_t yvel1,
    field_2d_t vol_flux_y)
{
    VOL_FLUX_Y(vol_flux_y, j, k) =
        0.25 * dt * YAREA(yarea, j, k)
        * (YVEL0(yvel0, j, k)
           + YVEL0(yvel0, j + 1, k)
           + YVEL1(yvel1, j, k)
           + YVEL1(yvel1, j + 1, k));
}
