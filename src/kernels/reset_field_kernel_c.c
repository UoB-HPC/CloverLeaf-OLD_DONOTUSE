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
 *  @brief C reset field kernel.
 *  @author Wayne Gaudin
 *  @details Copies all of the final end of step filed data to the begining of
 *  step data, ready for the next timestep.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void reset_field_kernel_c_(
    int x_min, int x_max, int y_min, int y_max,
    double * __restrict__ density0,
    const double * __restrict__ density1,
    double * __restrict__ energy0,
    const double * __restrict__ energy1,
    double * __restrict__ xvel0,
    const double * __restrict__ xvel1,
    double * __restrict__ yvel0,
    const double * __restrict__ yvel1)
{
    DOUBLEFOR(y_min, y_max, x_min, x_max, ({
        density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = density1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
        energy0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = energy1[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)];
    }));

    DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, ({
        xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = xvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
    }));

    DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, ({
        yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = yvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
    }));
}
