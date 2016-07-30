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
 *@brief C reset field kernel.
 *@author Wayne Gaudin
 *@details Copies all of the final end of step filed data to the begining of
 *  step data, ready for the next timestep.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

#define INRANGE(y, x, ymin, ymax, xmin, xmax) \
 ((y) >= (ymin) && (y) <= (ymax) && (x) >= (xmin) && (x) <= (xmax))

void reset_field_kernel_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t       density0,
    const_field_2d_t density1,
    field_2d_t       energy0,
    const_field_2d_t energy1,
    field_2d_t       xvel0,
    const_field_2d_t xvel1,
    field_2d_t       yvel0,
    const_field_2d_t yvel1)
{
    if (INRANGE(k, j, y_min, y_max, x_min, x_max)) {
        DENSITY0(density0, j, k) = DENSITY1(density1, j, k);
        ENERGY0(energy0, j, k) = ENERGY1(energy1, j, k);
    }

    XVEL0(xvel0, j, k) = XVEL1(xvel1, j, k);
    YVEL0(yvel0, j, k) = YVEL1(yvel1, j, k);
}
