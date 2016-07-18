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
 *@brief Driver for chunk initialisation.
 *@author Wayne Gaudin
 *@details Invokes the user specified chunk initialisation kernel.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void initialise_chunk_kernel_c_(
    int x_min, int x_max, int y_min, int y_max,
    double min_x,
    double min_y,
    double d_x,
    double d_y,
    field_1d_t vertexx,
    field_1d_t vertexdx,
    field_1d_t vertexy,
    field_1d_t vertexdy,
    field_1d_t cellx,
    field_1d_t celldx,
    field_1d_t celly,
    field_1d_t celldy,
    field_2d_t volume,
    field_2d_t xarea,
    field_2d_t yarea)
{
    int j, k;

#pragma ivdep
    for (j = x_min - 2; j <= x_max + 3; j++) {
        FIELD_1D(vertexx, j,  x_min - 2) = min_x + d_x * (double)(j - x_min);
    }


#pragma ivdep
    for (j = x_min - 2; j <= x_max + 3; j++) {
        FIELD_1D(vertexdx, j,  x_min - 2) = d_x;
    }


#pragma ivdep
    for (k = y_min - 2; k <= y_max + 3; k++) {
        FIELD_1D(vertexy, k,  y_min - 2) = min_y + d_y * (double)(k - y_min);
    }


#pragma ivdep
    for (k = y_min - 2; k <= y_max + 3; k++) {
        FIELD_1D(vertexdy, k,  y_min - 2) = d_y;
    }


#pragma ivdep
    for (j = x_min - 2; j <= x_max + 2; j++) {
        FIELD_1D(cellx, j,  x_min - 2) = 0.5 * (FIELD_1D(vertexx, j,  x_min - 2) + FIELD_1D(vertexx, j + 1,  x_min - 2));
    }


#pragma ivdep
    for (j = x_min - 2; j <= x_max + 2; j++) {
        FIELD_1D(celldx, j,  x_min - 2) = d_x;
    }


#pragma ivdep
    for (k = y_min - 2; k <= y_max + 2; k++) {
        FIELD_1D(celly, k,  y_min - 2) = 0.5 * (FIELD_1D(vertexy, k,  y_min - 2) + FIELD_1D(vertexy, k + 1,  x_min - 2));
    }


#pragma ivdep
    for (k = y_min - 2; k <= y_max + 2; k++) {
        FIELD_1D(celldy, k,  y_min - 2) = d_y;
    }


    for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
            VOLUME(volume, j, k) = d_x * d_y;
        }
    }


    for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
            XAREA(xarea, j, k) = FIELD_1D(celldy, k,  y_min - 2);
        }
    }


    for (k = y_min - 2; k <= y_max + 2; k++) {
#pragma ivdep
        for (j = x_min - 2; j <= x_max + 2; j++) {
            YAREA(yarea, j, k) = FIELD_1D(celldx, j,  x_min - 2);
        }
    }
}
