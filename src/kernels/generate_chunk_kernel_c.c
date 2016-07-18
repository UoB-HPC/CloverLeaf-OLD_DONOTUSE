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
 *@brief C mesh chunk generator
 *@author Wayne Gaudin
 *@details Generates the field data on a mesh chunk based on the user specified
 *  input for the states.
 *
 *  Note that state one is always used as the background state, which is then
 *  overwritten by further state definitions.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"


void generate_chunk_1_kernel(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t energy0,
    field_2d_t density0,
    field_2d_t xvel0,
    field_2d_t yvel0,
    const double* __restrict__ state_energy,
    const double* __restrict__ state_density,
    const double* __restrict__ state_xvel,
    const double* __restrict__ state_yvel)
{
    ENERGY0(energy0, j, k) = state_energy[FTNREF1D(1, 1)];
    DENSITY0(density0, j, k) = state_density[FTNREF1D(1, 1)];
    XVEL0(xvel0, j, k) = state_xvel[FTNREF1D(1, 1)];
    YVEL0(yvel0, j, k) = state_yvel[FTNREF1D(1, 1)];
}



void generate_chunk_kernel_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    double x_cent, double y_cent,
    field_1d_t vertexx,
    field_1d_t vertexy,
    field_1d_t cellx,
    field_1d_t celly,
    field_2d_t density0,
    field_2d_t energy0,
    field_2d_t xvel0,
    field_2d_t yvel0,
    int number_of_states,
    int state,
    double const* __restrict__ state_density,
    double const* __restrict__ state_energy,
    double const* __restrict__ state_xvel,
    double const* __restrict__ state_yvel,
    double const* __restrict__ state_xmin,
    double const* __restrict__ state_xmax,
    double const* __restrict__ state_ymin,
    double const* __restrict__ state_ymax,
    double const* __restrict__ state_radius,
    int* state_geometry)
{
    /* Could the velocity setting be thread unsafe? */
    if (state_geometry[FTNREF1D(state, 1)] == g_rect) {
        if (FIELD_1D(vertexx, j + 1,  x_min - 2) >= state_xmin[FTNREF1D(state, 1)] && FIELD_1D(vertexx, j,  x_min - 2) < state_xmax[FTNREF1D(state, 1)]) {
            if (FIELD_1D(vertexy, k + 1,  y_min - 2) >= state_ymin[FTNREF1D(state, 1)] && FIELD_1D(vertexy, k,  y_min - 2) < state_ymax[FTNREF1D(state, 1)]) {
                DENSITY0(density0, j, k) = state_density[FTNREF1D(state, 1)];
                ENERGY0(energy0, j, k) = state_energy[FTNREF1D(state, 1)];
                for (int kt = k; kt <= k + 1; kt++) {
                    for (int jt = j; jt <= j + 1; jt++) {
                        XVEL0(xvel0, jt, kt) = state_xvel[FTNREF1D(state, 1)];
                        YVEL0(yvel0, jt, kt) = state_yvel[FTNREF1D(state, 1)];
                    }
                }
            }
        }
    } else if (state_geometry[FTNREF1D(state, 1)] == g_circ) {
        double radius = sqrt((FIELD_1D(cellx, j,  x_min - 2) - x_cent) * (FIELD_1D(cellx, j,  x_min - 2) - x_cent) + (FIELD_1D(celly, k,  y_min - 2) - y_cent) * (FIELD_1D(celly, k,  y_min - 2) - y_cent));
        if (radius <= state_radius[FTNREF1D(state, 1)]) {
            DENSITY0(density0, j, k) = state_density[FTNREF1D(state, 1)];
            ENERGY0(energy0, j, k) = state_density[FTNREF1D(state, 1)];
            for (int kt = k; kt <= k + 1; kt++) {
                for (int jt = j; jt <= j + 1; jt++) {
                    XVEL0(xvel0, jt, kt) = state_xvel[FTNREF1D(state, 1)];
                    YVEL0(yvel0, jt, kt) = state_yvel[FTNREF1D(state, 1)];
                }
            }
        }
    } else if (state_geometry[FTNREF1D(state, 1)] == g_point) {
        if (FIELD_1D(vertexx, j,  x_min - 2) == x_cent && FIELD_1D(vertexy, j,  x_min - 2) == y_cent) {
            DENSITY0(density0, j, k) = state_density[FTNREF1D(state, 1)];
            ENERGY0(energy0, j, k) = state_density[FTNREF1D(state, 1)];
            for (int kt = k; kt <= k + 1; kt++) {
                for (int jt = j; jt <= j + 1; jt++) {
                    XVEL0(xvel0, jt, kt) = state_xvel[FTNREF1D(state, 1)];
                    YVEL0(yvel0, jt, kt) = state_yvel[FTNREF1D(state, 1)];
                }
            }
        }
    }
}
