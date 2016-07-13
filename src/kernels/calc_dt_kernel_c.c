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
 *  @brief C timestep kernel
 *  @author Wayne Gaudin
 *  @details calculates the minimum timestep on the mesh chunk based on the CFL
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
    const double * __restrict__ xarea     ,
    const double * __restrict__ yarea     ,
    const double * __restrict__ celldx    ,
    const double * __restrict__ celldy    ,
    const double * __restrict__ volume    ,
    const double * __restrict__ density0  ,
    const double * __restrict__ energy0   ,
    const double * __restrict__ pressure  ,
    const double * __restrict__ viscosity ,
    const double * __restrict__ soundspeed,
    const double * __restrict__ xvel0     ,
    const double * __restrict__ yvel0     ,
    double * __restrict__ dt_min)
{
    double dsx = celldx[FTNREF1D(j, x_min - 2)];
    double dsy = celldy[FTNREF1D(k, y_min - 2)];

    double cc = soundspeed[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] * soundspeed[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
    cc = cc + 2.0 * viscosity[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / density0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
    cc = MAX(sqrt(cc), g_small);

    double dtct = dtc_safe * MIN(dsx, dsy) / cc;

    double div = 0.0;

    double dv1 = (xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]) * xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
    double dv2 = (xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)] + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]) * xarea[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)];

    div = div + dv2 - dv1;

    double dtut = dtu_safe * 2.0 * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / MAX(fabs(dv1), MAX(fabs(dv2), g_small * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]));

    dv1 = (yvel0[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] + yvel0[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)]) * yarea[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
    dv2 = (yvel0[FTNREF2D(j, k + 1, x_max + 5, x_min - 2, y_min - 2)] + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]) * yarea[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)];

    div = div + dv2 - dv1;

    double dtvt = dtv_safe * 2.0 * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / MAX(fabs(dv1), MAX(fabs(dv2), g_small * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]));

    div = div / (2.0 * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]);
    double dtdivt;
    if (div < -g_small) {
        dtdivt = dtdiv_safe * (-1.0 / div);
    } else {
        dtdivt = g_big;
    }

    dt_min[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] =
        MIN(dtct, MIN(dtut, MIN(dtvt, dtdivt)));
}

void calc_dt_min_val(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    double * __restrict__ dt_min,
    double *dt_min_val)
{
    *dt_min_val = g_big;
    // TODO min reduction
    for (int k = y_min; k <= y_max; k++) {
#pragma ivdep
        for (int j = x_min; j <= x_max; j++) {
            if (dt_min[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] < *dt_min_val)
                *dt_min_val = dt_min[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
    }
}