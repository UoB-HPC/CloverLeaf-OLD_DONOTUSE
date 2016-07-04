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

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>

void calc_dt_kernel_c_(
    struct tile_type *tile,
    double g_small,
    double g_big,
    double min_dt,
    double dtc_safe,
    double dtu_safe,
    double dtv_safe,
    double dtdiv_safe,
    double *dt_min_val,
    int *dtlcontrol,
    double *xl_pos,
    double *yl_pos,
    int *j_ldt,
    int *k_ldt,
    int small)
{

    int x_min = tile->t_xmin,
        x_max = tile->t_xmax,
        y_min = tile->t_ymin,
        y_max = tile->t_ymax;

    double *xarea = tile->field.xarea;
    double *yarea = tile->field.yarea;
    double *cellx = tile->field.cellx;
    double *celly = tile->field.celly;
    double *celldx = tile->field.celldx;
    double *celldy = tile->field.celldy;
    double *volume = tile->field.volume;
    double *density0 = tile->field.density0;
    double *energy0 = tile->field.energy0;
    double *pressure = tile->field.pressure;
    double *viscosity = tile->field.viscosity;
    double *soundspeed = tile->field.soundspeed;
    double *xvel0 = tile->field.xvel0;
    double *yvel0 = tile->field.yvel0;
    double *dt_min = tile->field.work_array1;

    int j, k;

    double div, dsx, dsy, dtut, dtvt, dtct, dtdivt, cc, dv1, dv2, jk_control;

    small = 0;

    *dt_min_val = g_big;
    jk_control = 1.1;


    for (k = y_min; k <= y_max; k++) {
#pragma ivdep
        for (j = x_min; j <= x_max; j++) {

            dsx = celldx[FTNREF1D(j, x_min - 2)];
            dsy = celldy[FTNREF1D(k, y_min - 2)];

            cc = soundspeed[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] * soundspeed[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
            cc = cc + 2.0 * viscosity[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / density0[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
            cc = MAX(sqrt(cc), g_small);

            dtct = dtc_safe * MIN(dsx, dsy) / cc;

            div = 0.0;

            dv1 = (xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]) * xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)];
            dv2 = (xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)] + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]) * xarea[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)];

            div = div + dv2 - dv1;

            dtut = dtu_safe * 2.0 * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / MAX(fabs(dv1), MAX(fabs(dv2), g_small * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]));

            dv1 = (yvel0[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] + yvel0[FTNREF2D(j + 1, k, x_max + 5, x_min - 2, y_min - 2)]) * yarea[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)];
            dv2 = (yvel0[FTNREF2D(j, k + 1, x_max + 5, x_min - 2, y_min - 2)] + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]) * yarea[FTNREF2D(j, k + 1, x_max + 4, x_min - 2, y_min - 2)];

            div = div + dv2 - dv1;

            dtvt = dtv_safe * 2.0 * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)] / MAX(fabs(dv1), MAX(fabs(dv2), g_small * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]));

            div = div / (2.0 * volume[FTNREF2D(j, k, x_max + 4, x_min - 2, y_min - 2)]);

            if (div < -g_small) {
                dtdivt = dtdiv_safe * (-1.0 / div);
            } else {
                dtdivt = g_big;
            }

            dt_min[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] = MIN(dtct, MIN(dtut, MIN(dtvt, dtdivt)));

        }
    }


    for (k = y_min; k <= y_max; k++) {
#pragma ivdep
        for (j = x_min; j <= x_max; j++) {
            if (dt_min[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)] < *dt_min_val) *dt_min_val = dt_min[FTNREF2D(j, k, x_max + 5, x_min - 2, y_min - 2)];
        }
    }



    // Extract the mimimum timestep information
    // dtl_control = 10.01 * (jk_control - (int)(jk_control));
    jk_control = jk_control - (jk_control - (int)(jk_control));
    *j_ldt = 1; //MOD(INT(jk_control),x_max)
    *k_ldt = 1; //1+(jk_control/x_max)
    //xl_pos=cellx[FTNREF1D(j_ldt,xmin-2)];
    //yl_pos=celly[FTNREF1D(j_ldt,ymin-2)];

    if (*dt_min_val < min_dt) small = 1;

    *dtlcontrol = 1;

    if (small != 0) {
        printf("Timestep information:\n");
        printf("j, k                 :%i %i \n", *j_ldt, *k_ldt);
        printf("x, y                 :%f %f \n", *xl_pos, *yl_pos);
        printf("timestep : %f\n", *dt_min_val);
        printf("Cell velocities;\n");
        printf("%f %f \n", xvel0[FTNREF2D(*j_ldt  , *k_ldt  , x_max + 5, x_min - 2, y_min - 2)], yvel0[FTNREF2D(*j_ldt  , *k_ldt  , x_max + 5, x_min - 2, y_min - 2)]);
        printf("%f %f \n", xvel0[FTNREF2D(*j_ldt + 1, *k_ldt  , x_max + 5, x_min - 2, y_min - 2)], yvel0[FTNREF2D(*j_ldt + 1, *k_ldt  , x_max + 5, x_min - 2, y_min - 2)]);
        printf("%f %f \n", xvel0[FTNREF2D(*j_ldt + 1, *k_ldt + 1, x_max + 5, x_min - 2, y_min - 2)], yvel0[FTNREF2D(*j_ldt + 1, *k_ldt + 1, x_max + 5, x_min - 2, y_min - 2)]);
        printf("%f %f \n", xvel0[FTNREF2D(*j_ldt  , *k_ldt + 1, x_max + 5, x_min - 2, y_min - 2)], yvel0[FTNREF2D(*j_ldt  , *k_ldt + 1, x_max + 5, x_min - 2, y_min - 2)]);
        printf("density, energy, pressure, soundspeed \n");
        printf("%f %f %f %f \n", density0[FTNREF2D(*j_ldt, *k_ldt, x_max + 4, x_min - 2, y_min - 2)], energy0[FTNREF2D(*j_ldt, *k_ldt, x_max + 4, x_min - 2, y_min - 2)], pressure[FTNREF2D(*j_ldt, *k_ldt, x_max + 4, x_min - 2, y_min - 2)], soundspeed[FTNREF2D(*j_ldt, *k_ldt, x_max + 4, x_min - 2, y_min - 2)]);
    }

}
