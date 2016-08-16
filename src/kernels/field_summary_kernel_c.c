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
 *@brief C field summary kernel
 *@author Wayne Gaudin
 *@details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

kernelqual void field_summary_kernel(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t volume,
    const_field_2d_t density0,
    const_field_2d_t energy0,
    const_field_2d_t pressure,
    const_field_2d_t xvel0,
    const_field_2d_t yvel0,
    double* vol,
    double* mass,
    double* ie,
    double* ke,
    double* press)
{
    double vsqrd = 0.0;
    for (int kv = k; kv <= k + 1; kv++) {
        for (int jv = j; jv <= j + 1; jv++) {
            vsqrd += 0.25 * (XVEL0(xvel0, jv , kv) * XVEL0(xvel0, jv , kv)
                             + YVEL0(yvel0, jv , kv) * YVEL0(yvel0, jv , kv));
        }
    }
    double cell_vol = VOLUME(volume, j, k);
    double cell_mass = cell_vol * DENSITY0(density0, j, k);

    *vol   += cell_vol;
    *mass  += cell_mass;
    *ie    += cell_mass * ENERGY0(energy0, j, k);
    *ke    += cell_mass * 0.5 * vsqrd;
    *press += cell_vol * PRESSURE(pressure, j, k);
}
