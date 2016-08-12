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
 *@brief C PdV kernel.
 *@author Wayne Gaudin
 *@details Calculates the change in energy and density in a cell using the
 *  change on cell volume due to the velocity gradients in a cell. The time
 *  level of the velocity data depends on whether it is invoked as the
 *  predictor or corrector.
 */

// #include <stdio.h>
// #include <stdlib.h>
// #include "ftocmacros.h"
// #include <math.h>
// #include "../definitions_c.h"


kernelqual void pdv_kernel_predict_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t yarea,
    const_field_2d_t volume,
    const_field_2d_t density0,
    field_2d_t       density1,
    const_field_2d_t energy0,
    field_2d_t       energy1,
    const_field_2d_t pressure,
    const_field_2d_t viscosity,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    const_field_2d_t yvel0,
    const_field_2d_t yvel1,
    field_2d_t       volume_change)
{
    double left_flux = (XAREA(xarea, j, k))
                       * (XVEL0(xvel0, j, k)
                          + XVEL0(xvel0, j, k + 1)
                          + XVEL0(xvel0, j, k)
                          + XVEL0(xvel0, j, k + 1))
                       * 0.25 * dt * 0.5;
    double right_flux = (XAREA(xarea, j + 1, k))
                        * (XVEL0(xvel0, j + 1, k)
                           + XVEL0(xvel0, j + 1, k + 1)
                           + XVEL0(xvel0, j + 1, k)
                           + XVEL0(xvel0, j + 1, k + 1))
                        * 0.25 * dt * 0.5;
    double bottom_flux = (YAREA(yarea, j, k))
                         * (YVEL0(yvel0, j, k)
                            + YVEL0(yvel0, j + 1, k)
                            + YVEL0(yvel0, j, k)
                            + YVEL0(yvel0, j + 1, k))
                         * 0.25 * dt * 0.5;
    double top_flux = (YAREA(yarea, j, k + 1))
                      * (YVEL0(yvel0, j, k + 1)
                         + YVEL0(yvel0, j + 1, k + 1)
                         + YVEL0(yvel0, j, k + 1)
                         + YVEL0(yvel0, j + 1, k + 1))
                      * 0.25 * dt * 0.5;

    double total_flux = right_flux - left_flux + top_flux - bottom_flux;

    WORK_ARRAY(volume_change, j, k) = VOLUME(volume, j, k)
                                      / (VOLUME(volume, j, k) + total_flux);

// min_cell_volume = MIN(VOLUME(volume, j,k) + right_flux - left_flux + top_flux - bottom_flux
// , MIN(VOLUME(volume, j,k) + right_flux - left_flux
// , VOLUME(volume, j,k) + top_flux - bottom_flux));

    double recip_volume = 1.0 / VOLUME(volume, j, k);

    double energy_change = (PRESSURE(pressure, j, k) / DENSITY0(density0, j, k)
                            + VISCOSITY(viscosity, j, k) / DENSITY0(density0, j, k))
                           * total_flux * recip_volume;

    ENERGY1(energy1, j, k) = ENERGY0(energy0, j, k) - energy_change;

    DENSITY1(density1, j, k) = DENSITY0(density0, j, k)
                               * WORK_ARRAY(volume_change, j, k);
}

kernelqual void pdv_kernel_no_predict_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t yarea,
    const_field_2d_t volume,
    const_field_2d_t density0,
    field_2d_t       density1,
    const_field_2d_t energy0,
    field_2d_t       energy1,
    const_field_2d_t pressure,
    const_field_2d_t viscosity,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    const_field_2d_t yvel0,
    const_field_2d_t yvel1,
    field_2d_t       volume_change)
{
    double left_flux = (XAREA(xarea, j, k))
                       * (XVEL0(xvel0, j, k)
                          + XVEL0(xvel0, j, k + 1)
                          + XVEL1(xvel1, j, k)
                          + XVEL1(xvel1, j, k + 1))
                       * 0.25 * dt;
    double right_flux = (XAREA(xarea, j + 1, k))
                        * (XVEL0(xvel0, j + 1, k)
                           + XVEL0(xvel0, j + 1, k + 1)
                           + XVEL1(xvel1, j + 1, k)
                           + XVEL1(xvel1, j + 1, k + 1))
                        * 0.25 * dt;
    double bottom_flux = (YAREA(yarea, j, k))
                         * (YVEL0(yvel0, j, k)
                            + YVEL0(yvel0, j + 1, k)
                            + YVEL1(yvel1, j, k)
                            + YVEL1(yvel1, j + 1, k))
                         * 0.25 * dt;
    double top_flux = (YAREA(yarea, j, k + 1))
                      * (YVEL0(yvel0, j, k + 1)
                         + YVEL0(yvel0, j + 1, k + 1)
                         + YVEL1(yvel1, j, k + 1)
                         + YVEL1(yvel1, j + 1, k + 1))
                      * 0.25 * dt;

    double total_flux = right_flux - left_flux + top_flux - bottom_flux;

    WORK_ARRAY(volume_change, j, k) = VOLUME(volume, j, k)
                                      / (VOLUME(volume, j, k) + total_flux);

// min_cell_volume = MIN(VOLUME(volume, j,k) + right_flux - left_flux + top_flux - bottom_flux
// , MIN(VOLUME(volume, j,k) + right_flux - left_flux
// , VOLUME(volume, j,k) + top_flux - bottom_flux));

    double recip_volume = 1.0 / VOLUME(volume, j, k);

    double energy_change = (PRESSURE(pressure, j, k) / DENSITY0(density0, j, k)
                            + VISCOSITY(viscosity, j, k) / DENSITY0(density0, j, k))
                           * total_flux * recip_volume;

    ENERGY1(energy1, j, k) = ENERGY0(energy0, j, k) - energy_change;

    DENSITY1(density1, j, k) = DENSITY0(density0, j, k)
                               * WORK_ARRAY(volume_change, j, k);
}

