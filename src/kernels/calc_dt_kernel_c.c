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
 *@brief C timestep kernel
 *@author Wayne Gaudin
 *@details calculates the minimum timestep on the mesh chunk based on the CFL
 *  condition, the velocity gradient and the velocity divergence. A safety
 *  factor is used to ensure numerical stability.
 */

// #include <stdio.h>
// #include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void calc_dt_kernel_c_(
    int j, int k,
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    const_field_2d_t xarea,
    const_field_2d_t yarea,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    const_field_2d_t volume,
    const_field_2d_t density0,
    const_field_2d_t energy0 ,
    const_field_2d_t pressure,
    const_field_2d_t viscosity,
    const_field_2d_t soundspeed,
    const_field_2d_t xvel0,
    const_field_2d_t yvel0,
    field_2d_t dtmin)
{
    double dsx = FIELD_1D(celldx, j,  x_min - 2);
    double dsy = FIELD_1D(celldy, k,  y_min - 2);

    double cc = SOUNDSPEED(soundspeed, j, k) * SOUNDSPEED(soundspeed, j, k);
    cc = cc + 2.0 * VISCOSITY(viscosity, j, k) / DENSITY0(density0, j, k);
    cc = MAX(sqrt(cc), g_small);

    double dtct = dtc_safe * MIN(dsx, dsy) / cc;

    double div = 0.0;

    double dv1 = (XVEL0(xvel0, j, k) + XVEL0(xvel0, j, k + 1)) * XAREA(xarea, j, k);
    double dv2 = (XVEL0(xvel0, j + 1, k) + XVEL0(xvel0, j + 1, k + 1)) * XAREA(xarea, j + 1, k);

    div = div + dv2 - dv1;

    double dtut = dtu_safe * 2.0 * VOLUME(volume, j, k) / MAX(fabs(dv1), MAX(fabs(dv2), g_small * VOLUME(volume, j, k)));

    dv1 = (YVEL0(yvel0, j, k) + YVEL0(yvel0, j + 1, k)) * YAREA(yarea, j, k);
    dv2 = (YVEL0(yvel0, j, k + 1) + YVEL0(yvel0, j + 1, k + 1)) * YAREA(yarea, j, k + 1);

    div = div + dv2 - dv1;

    double dtvt = dtv_safe * 2.0 * VOLUME(volume, j, k) / MAX(fabs(dv1), MAX(fabs(dv2), g_small * VOLUME(volume, j, k)));

    div = div / (2.0 * VOLUME(volume, j, k));
    double dtdivt;
    if (div < -g_small) {
        dtdivt = dtdiv_safe * (-1.0 / div);
    } else {
        dtdivt = g_big;
    }

    WORK_ARRAY(dtmin, j, k) = MIN(dtct, MIN(dtut, MIN(dtvt, dtdivt)));
}


void calc_dt_min_val(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    field_2d_t dt_min,
    double* dt_min_val)
{
#if defined(USE_KOKKOS)

    *dt_min_val = g_big;
    for (int k = y_min; k <= y_max; k++) {
// #pragma ivdep
        for (int j = x_min; j <= x_max; j++) {
            if (WORK_ARRAY(dt_min, j,  k) < *dt_min_val)
                *dt_min_val = WORK_ARRAY(dt_min, j,  k);
        }
    }

#else

    double minval = g_big;
    #pragma omp parallel for reduction(min:minval)
    for (int k = y_min; k <= y_max; k++) {
        for (int j = x_min; j <= x_max; j++) {
            if (WORK_ARRAY(dt_min, j,  k) < minval)
                minval = WORK_ARRAY(dt_min, j,  k);
        }
    }
    *dt_min_val = minval;

#endif
}
