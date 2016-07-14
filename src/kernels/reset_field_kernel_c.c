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

void reset_field_kernel_c_(
    int x_min, int x_max, int y_min, int y_max,
    FIELDPARAM density0,
    CONSTFIELDPARAM density1,
    FIELDPARAM energy0,
    CONSTFIELDPARAM energy1,
    FIELDPARAM xvel0,
    CONSTFIELDPARAM xvel1,
    FIELDPARAM yvel0,
    CONSTFIELDPARAM yvel1)
{
    DOUBLEFOR(y_min, y_max, x_min, x_max, ({
        DENSITY0(density0, j, k) = DENSITY1(density1, j, k);
        ENERGY0(energy0, j, k) = ENERGY1(energy1, j, k);
    }));

    DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, ({
        XVEL0(xvel0, j, k) = XVEL1(xvel1, j, k);
    }));

    DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, ({
        YVEL0(yvel0, j, k) = YVEL1(yvel1, j, k);
    }));
}
