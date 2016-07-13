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
 *@brief C acceleration kernel
 *@author Wayne Gaudin
 *@details The pressure and viscosity gradients are used to update the
 *  velocity field.
 */

#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void accelerate_kernel_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    const double* __restrict__ xarea,
    const double* __restrict__ yarea,
    const double* __restrict__ volume,
    const double* __restrict__ density0 ,
    const double* __restrict__ pressure ,
    const double* __restrict__ viscosity,
    double* __restrict__ xvel0,
    double* __restrict__ yvel0,
    double* __restrict__ xvel1,
    double* __restrict__ yvel1,
    double dt)
{
    double nodal_mass = (DENSITY0(density0, j - 1, k - 1) * VOLUME(volume, j - 1, k - 1)
                         + DENSITY0(density0, j, k - 1) * VOLUME(volume, j, k - 1)
                         + DENSITY0(density0, j, k) * VOLUME(volume, j, k)
                         + DENSITY0(density0, j - 1, k) * VOLUME(volume, j - 1, k))
                        * 0.25;
    double stepby_mass_s = 0.5 * dt / nodal_mass;
    XVEL1(xvel1, j, k) = XVEL0(xvel0, j, k)
                         - stepby_mass_s
                         * (XAREA(xarea, j, k)
                            * (PRESSURE(pressure, j, k) - PRESSURE(pressure, j - 1, k))
                            + XAREA(xarea, j, k - 1)
                            * (PRESSURE(pressure, j, k - 1) - PRESSURE(pressure, j - 1, k - 1)));

    YVEL1(yvel1, j, k) = YVEL0(yvel0, j, k)
                         - stepby_mass_s
                         * (YAREA(yarea, j, k)
                            * (PRESSURE(pressure, j, k) - PRESSURE(pressure, j, k - 1))
                            + YAREA(yarea, j - 1, k)
                            * (PRESSURE(pressure, j - 1, k) - PRESSURE(pressure, j - 1, k - 1)));

    XVEL1(xvel1, j, k) = XVEL1(xvel1, j, k)
                         - stepby_mass_s
                         * (XAREA(xarea, j, k)
                            * (VISCOSITY(viscosity, j, k) - VISCOSITY(viscosity, j - 1, k))
                            + XAREA(xarea, j, k - 1)
                            * (VISCOSITY(viscosity, j, k - 1) - VISCOSITY(viscosity, j - 1, k - 1)));

    YVEL1(yvel1, j, k) = YVEL1(yvel1, j, k)
                         - stepby_mass_s
                         * (YAREA(yarea, j, k)
                            * (VISCOSITY(viscosity, j, k) - VISCOSITY(viscosity, j, k - 1))
                            + YAREA(yarea, j - 1, k)
                            * (VISCOSITY(viscosity, j - 1, k) - VISCOSITY(viscosity, j - 1, k - 1)));
}
