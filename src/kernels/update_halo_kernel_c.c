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

// #include <stdio.h>
// #include <stdlib.h>
// #include "ftocmacros.h"
// #include <math.h>


/* These need to be kept consistent with the data module to avoid use statement */
#define CHUNK_LEFT     1
#define CHUNK_RIGHT    2
#define CHUNK_BOTTOM   3
#define CHUNK_TOP      4
#define EXTERNAL_FACE  -1
#define TILE_LEFT      1
#define TILE_RIGHT     2
#define TILE_BOTTOM    3
#define TILE_TOP       4
#define EXTERNAL_TILE  -1

#define FIELD_DENSITY0     1
#define FIELD_DENSITY1     2
#define FIELD_ENERGY0      3
#define FIELD_ENERGY1      4
#define FIELD_PRESSURE     5
#define FIELD_VISCOSITY    6
#define FIELD_SOUNDSPEED   7
#define FIELD_XVEL0        8
#define FIELD_XVEL1        9
#define FIELD_YVEL0        10
#define FIELD_YVEL1        11
#define FIELD_VOL_FLUX_X   12
#define FIELD_VOL_FLUX_Y   13
#define FIELD_MASS_FLUX_X  14
#define FIELD_MASS_FLUX_Y  15


// requires fields, chunk_neighbours, tile_neighbours, x_min,x_max,y_min,y_max
// depth to be defined
#define update1(j, k, field_t, field, access) \
    if (fields[FTNREF1D(field_t, 1)] == 1) { \
        if (chunk_neighbours[FTNREF1D(CHUNK_BOTTOM, 1)] == EXTERNAL_FACE && \
            tile_neighbours[FTNREF1D(TILE_BOTTOM, 1)] == EXTERNAL_TILE) { \
            access(field, j, 1 - k) = \
                access(field, j, 0 + k); \
        } \
        if (chunk_neighbours[FTNREF1D(CHUNK_TOP, 1)] == EXTERNAL_FACE && \
            tile_neighbours[FTNREF1D(TILE_TOP, 1)] == EXTERNAL_TILE) { \
            access(field, j, y_max + k) = \
                access(field, j, y_max + 1 - k); \
        } \
    }

#define update2(j, k, field_t, field, access) \
    if (fields[FTNREF1D(field_t, 1)] == 1) { \
        if (chunk_neighbours[FTNREF1D(CHUNK_LEFT, 1)] == EXTERNAL_FACE && \
            tile_neighbours[FTNREF1D(TILE_LEFT, 1)] == EXTERNAL_TILE) { \
            access(field, 1 - j, k) = \
                access(field, 0 + j, k); \
        } \
        if (chunk_neighbours[FTNREF1D(CHUNK_RIGHT, 1)] == EXTERNAL_FACE && \
            tile_neighbours[FTNREF1D(TILE_RIGHT, 1)] == EXTERNAL_TILE) { \
            access(field, x_max + j, k) = \
                access(field, x_max + 1 - j, k); \
        } \
    }

void update_halo_kernel_1(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    flag_t chunk_neighbours,
    flag_t tile_neighbours,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1,
    field_2d_t pressure,
    field_2d_t viscosity,
    field_2d_t soundspeed,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
    field_2d_t vol_flux_x,
    field_2d_t mass_flux_x,
    field_2d_t vol_flux_y,
    field_2d_t mass_flux_y,
    flag_t fields,
    int depth)
{
    update1(j, k, FIELD_DENSITY0,    density0,    DENSITY0);
    update1(j, k, FIELD_DENSITY1,    density1,    DENSITY1);
    update1(j, k, FIELD_ENERGY0,     energy0,     ENERGY0);
    update1(j, k, FIELD_ENERGY1,     energy1,     ENERGY1);
    update1(j, k, FIELD_PRESSURE,    pressure,    PRESSURE);
    update1(j, k, FIELD_VISCOSITY,   viscosity,   VISCOSITY);
    update1(j, k, FIELD_SOUNDSPEED,  soundspeed,  SOUNDSPEED);
    update1(j, k, FIELD_XVEL0,       xvel0,       XVEL0);
    update1(j, k, FIELD_XVEL1,       yvel0,       XVEL1);
    update1(j, k, FIELD_YVEL0,       xvel1,       YVEL0);
    update1(j, k, FIELD_YVEL1,       yvel1,       YVEL1);
    update1(j, k, FIELD_VOL_FLUX_X,  vol_flux_x,  VOL_FLUX_X);
    update1(j, k, FIELD_MASS_FLUX_X, mass_flux_x, MASS_FLUX_X);
    update1(j, k, FIELD_VOL_FLUX_Y,  vol_flux_y,  VOL_FLUX_Y);
    update1(j, k, FIELD_MASS_FLUX_Y, mass_flux_y, MASS_FLUX_Y);
}

void update_halo_kernel_2(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    flag_t chunk_neighbours,
    flag_t tile_neighbours,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1,
    field_2d_t pressure,
    field_2d_t viscosity,
    field_2d_t soundspeed,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
    field_2d_t vol_flux_x,
    field_2d_t mass_flux_x,
    field_2d_t vol_flux_y,
    field_2d_t mass_flux_y,
    flag_t fields,
    int depth)
{
    update2(j, k, FIELD_DENSITY0,    density0,    DENSITY0);
    update2(j, k, FIELD_DENSITY1,    density1,    DENSITY1);
    update2(j, k, FIELD_ENERGY0,     energy0,     ENERGY0);
    update2(j, k, FIELD_ENERGY1,     energy1,     ENERGY1);
    update2(j, k, FIELD_PRESSURE,    pressure,    PRESSURE);
    update2(j, k, FIELD_VISCOSITY,   viscosity,   VISCOSITY);
    update2(j, k, FIELD_SOUNDSPEED,  soundspeed,  SOUNDSPEED);
    update2(j, k, FIELD_XVEL0,       xvel0,       XVEL0);
    update2(j, k, FIELD_XVEL1,       yvel0,       XVEL1);
    update2(j, k, FIELD_YVEL0,       xvel1,       YVEL0);
    update2(j, k, FIELD_YVEL1,       yvel1,       YVEL1);
    update2(j, k, FIELD_VOL_FLUX_X,  vol_flux_x,  VOL_FLUX_X);
    update2(j, k, FIELD_MASS_FLUX_X, mass_flux_x, MASS_FLUX_X);
    update2(j, k, FIELD_VOL_FLUX_Y,  vol_flux_y,  VOL_FLUX_Y);
    update2(j, k, FIELD_MASS_FLUX_Y, mass_flux_y, MASS_FLUX_Y);
}