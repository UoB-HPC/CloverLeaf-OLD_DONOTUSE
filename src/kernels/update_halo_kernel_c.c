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
 *@brief C kernel to update the external halo cells in a chunk.
 *@author Wayne Gaudin
 *@details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>

void update_halo_kernel_c_(int* xmin, int* xmax, int* ymin, int* ymax,
                           int* chunk_neighbours,
                           int* tile_neighbours,
                           double* __restrict__ density0,
                           double* __restrict__ energy0,
                           double* __restrict__ pressure,
                           double* __restrict__ viscosity,
                           double* __restrict__ soundspeed,
                           double* __restrict__ density1,
                           double* __restrict__ energy1,
                           double* __restrict__ xvel0,
                           double* __restrict__ yvel0,
                           double* __restrict__ xvel1,
                           double* __restrict__ yvel1,
                           double* __restrict__ vol_flux_x,
                           double* __restrict__ vol_flux_y,
                           double* __restrict__ mass_flux_x,
                           double* mass_flux_y,
                           int* fields,
                           int* dpth)
{

    int x_min = *xmin;
    int x_max = *xmax;
    int y_min = *ymin;
    int y_max = *ymax;
    int depth = *dpth;

    /* These need to be kept consistent with the data module to avoid use statement */
    int CHUNK_LEFT = 1, CHUNK_RIGHT = 2, CHUNK_BOTTOM = 3, CHUNK_TOP = 4, EXTERNAL_FACE = -1;
    int TILE_LEFT = 1, TILE_RIGHT = 2, TILE_BOTTOM = 3, TILE_TOP = 4, EXTERNAL_TILE = -1;

    int FIELD_DENSITY0 = 1;
    int FIELD_DENSITY1 = 2;
    int FIELD_ENERGY0 = 3;
    int FIELD_ENERGY1 = 4;
    int FIELD_PRESSURE = 5;
    int FIELD_VISCOSITY = 6;
    int FIELD_SOUNDSPEED = 7;
    int FIELD_XVEL0 = 8;
    int FIELD_XVEL1 = 9;
    int FIELD_YVEL0 = 10;
    int FIELD_YVEL1 = 11;
    int FIELD_VOL_FLUX_X = 12;
    int FIELD_VOL_FLUX_Y = 13;
    int FIELD_MASS_FLUX_X = 14;
    int FIELD_MASS_FLUX_Y = 15;

    int j, k;

    /* Update values in external halo cells based on depth and fields requested */



    if (fields[FTNREF1D(FIELD_DENSITY0, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    DENSITY0(density0, j, 1 - k) = DENSITY0(density0, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    DENSITY0(density0, j, y_max + k) = DENSITY0(density0, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    DENSITY0(density0, 1 - j, k) = DENSITY0(density0, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    DENSITY0(density0, x_max + j, k) = DENSITY0(density0, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_DENSITY1, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    DENSITY1(density1, j, 1 - k) = DENSITY1(density1, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    DENSITY1(density1, j, y_max + k) = DENSITY1(density1, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    DENSITY1(density1, 1 - j, k) = DENSITY1(density1, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    DENSITY1(density1, x_max + j, k) = DENSITY1(density1, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_ENERGY0, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    ENERGY0(energy0, j, 1 - k) = ENERGY0(energy0, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    ENERGY0(energy0, j, y_max + k) = ENERGY0(energy0, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    ENERGY0(energy0, 1 - j, k) = ENERGY0(energy0, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    ENERGY0(energy0, x_max + j, k) = ENERGY0(energy0, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_ENERGY1, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    ENERGY1(energy1, j, 1 - k) = ENERGY1(energy1, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    ENERGY1(energy1, j, y_max + k) = ENERGY1(energy1, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    ENERGY1(energy1, 1 - j, k) = ENERGY1(energy1, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    ENERGY1(energy1, x_max + j, k) = ENERGY1(energy1, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_PRESSURE, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    PRESSURE(pressure, j, 1 - k) = PRESSURE(pressure, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    PRESSURE(pressure, j, y_max + k) = PRESSURE(pressure, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    PRESSURE(pressure, 1 - j, k) = PRESSURE(pressure, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    PRESSURE(pressure, x_max + j, k) = PRESSURE(pressure, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_VISCOSITY, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    VISCOSITY(viscosity, j, 1 - k) = VISCOSITY(viscosity, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    VISCOSITY(viscosity, j, y_max + k) = VISCOSITY(viscosity, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    VISCOSITY(viscosity, 1 - j, k) = VISCOSITY(viscosity, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    VISCOSITY(viscosity, x_max + j, k) = VISCOSITY(viscosity, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_SOUNDSPEED, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    SOUNDSPEED(soundspeed, j, 1 - k) = SOUNDSPEED(soundspeed, j, 0 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    SOUNDSPEED(soundspeed, j, y_max + k) = SOUNDSPEED(soundspeed, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    SOUNDSPEED(soundspeed, 1 - j, k) = SOUNDSPEED(soundspeed, 0 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    SOUNDSPEED(soundspeed, x_max + j, k) = SOUNDSPEED(soundspeed, x_max + 1 - j, k);
                }
            }
        }
    }
    if (fields[FTNREF1D(FIELD_XVEL0, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    XVEL0(xvel0, j, 1 - k) = XVEL0(xvel0, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    XVEL0(xvel0, j, y_max + 1 + k) = XVEL0(xvel0, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    XVEL0(xvel0, 1 - j, k) = -XVEL0(xvel0, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    XVEL0(xvel0, x_max + 1 + j, k) = -XVEL0(xvel0, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_XVEL1, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    XVEL1(xvel1, j, 1 - k) = XVEL1(xvel1, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    XVEL1(xvel1, j, y_max + 1 + k) = XVEL1(xvel1, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    XVEL1(xvel1, 1 - j, k) = -XVEL1(xvel1, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    XVEL1(xvel1, x_max + 1 + j, k) = -XVEL1(xvel1, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_YVEL0, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    YVEL0(yvel0, j, 1 - k) = -YVEL0(yvel0, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    YVEL0(yvel0, j, y_max + 1 + k) = -YVEL0(yvel0, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    YVEL0(yvel0, 1 - j, k) = YVEL0(yvel0, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    YVEL0(yvel0, x_max + 1 + j, k) = YVEL0(yvel0, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_YVEL1, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    YVEL1(yvel1, j, 1 - k) = -YVEL1(yvel1, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    YVEL1(yvel1, j, y_max + 1 + k) = -YVEL1(yvel1, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    YVEL1(yvel1, 1 - j, k) = YVEL1(yvel1, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    YVEL1(yvel1, x_max + 1 + j, k) = YVEL1(yvel1, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_VOL_FLUX_X, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    VOL_FLUX_X(vol_flux_x, j, 1 - k) = VOL_FLUX_X(vol_flux_x, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    VOL_FLUX_X(vol_flux_x, j, y_max + k) = VOL_FLUX_X(vol_flux_x, j, y_max - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    VOL_FLUX_X(vol_flux_x, 1 - j, k) = -VOL_FLUX_X(vol_flux_x, 1 + j, k);
                }
            }
        }
        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    VOL_FLUX_X(vol_flux_x, x_max + 1 + j, k) = -VOL_FLUX_X(vol_flux_x, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_MASS_FLUX_X, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    MASS_FLUX_X(mass_flux_x, j, 1 - k) = MASS_FLUX_X(mass_flux_x, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + 1 + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    MASS_FLUX_X(mass_flux_x, j, y_max + k) = MASS_FLUX_X(mass_flux_x, j, y_max - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    MASS_FLUX_X(mass_flux_x, 1 - j, k) = -MASS_FLUX_X(mass_flux_x, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    MASS_FLUX_X(mass_flux_x, x_max + 1 + j, k) = -MASS_FLUX_X(mass_flux_x, x_max + 1 - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_VOL_FLUX_Y, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    VOL_FLUX_Y(vol_flux_y, j, 1 - k) = -VOL_FLUX_Y(vol_flux_y, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    VOL_FLUX_Y(vol_flux_y, j, y_max + k + 1) = -VOL_FLUX_Y(vol_flux_y, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    VOL_FLUX_Y(vol_flux_y, 1 - j, k) = VOL_FLUX_Y(vol_flux_y, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    VOL_FLUX_Y(vol_flux_y, x_max + j, k) = VOL_FLUX_Y(vol_flux_y, x_max - j, k);
                }
            }
        }
    }

    if (fields[FTNREF1D(FIELD_MASS_FLUX_Y, 1)] == 1) {
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    MASS_FLUX_Y(mass_flux_y, j, 1 - k) = -MASS_FLUX_Y(mass_flux_y, j, 1 + k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) {

            for (j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
                for (k = 1; k <= depth; k++) {
                    MASS_FLUX_Y(mass_flux_y, j, y_max + k + 1) = -MASS_FLUX_Y(mass_flux_y, j, y_max + 1 - k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    MASS_FLUX_Y(mass_flux_y, 1 - j, k) = MASS_FLUX_Y(mass_flux_y, 1 + j, k);
                }
            }
        }

        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) {

            for (k = y_min - depth; k <= y_max + 1 + depth; k++) {
#pragma ivdep
                for (j = 1; j <= depth; j++) {
                    MASS_FLUX_Y(mass_flux_y, x_max + j, k) = MASS_FLUX_Y(mass_flux_y, x_max - j, k);
                }
            }
        }
    }



}
