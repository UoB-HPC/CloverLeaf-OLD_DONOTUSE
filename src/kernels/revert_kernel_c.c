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
 *@brief C revert kernel.
 *@author Wayne Gaudin
 *@details Takes the half step field data used in the predictor and reverts
 *  it to the start of step data, ready for the corrector.
 *  Note that this does not seem necessary in this proxy-app but should be
 *  left in to remain relevant to the full method.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>

void revert_kernel_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1)
{
    DENSITY1(density1, j, k) = DENSITY0(density0, j, k);
    ENERGY1(energy1, j, k) = ENERGY0(energy0, j, k);
}
