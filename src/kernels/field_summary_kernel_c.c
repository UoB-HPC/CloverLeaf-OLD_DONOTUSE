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
 *@brief C field summary kernel
 *@author Wayne Gaudin
 *@details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void field_summary_kernel_c_(int* xmin,
                             int* xmax,
                             int* ymin,
                             int* ymax,
                             field_2d_t volume,
                             field_2d_t density0,
                             field_2d_t energy0,
                             field_2d_t pressure,
                             field_2d_t xvel0,
                             field_2d_t yvel0,
                             double* __restrict__ vl,
                             double* __restrict__ mss,
                             double* __restrict__ ien,
                             double* __restrict__ ken,
                             double* __restrict__ prss)
{

    int x_min = *xmin;
    int x_max = *xmax;
    int y_min = *ymin;
    int y_max = *ymax;
    double vol = *vl;
    double mass = *mss;
    double ie = *ien;
    double ke = *ken;
    double press = *prss;

    int j, k, jv, kv;


    vol = 0.0;
    mass = 0.0;
    ie = 0.0;
    ke = 0.0;;
    press = 0.0;



// TODO reduction
    #pragma omp parallel for reduction(+:vol, mass, ie, ke, press)
    for (k = y_min; k <= y_max; k++) {
#pragma ivdep
        for (j = x_min; j <= x_max; j++) {
            double vsqrd = 0.0;
            for (kv = k; kv <= k + 1; kv++) {
                for (jv = j; jv <= j + 1; jv++) {
                    vsqrd = vsqrd + 0.25 * (XVEL0(xvel0, jv , kv) * XVEL0(xvel0, jv , kv)
                                            + YVEL0(yvel0, jv , kv) * YVEL0(yvel0, jv , kv));
                }
            }
            double cell_vol = VOLUME(volume, j, k);
            double cell_mass = cell_vol * DENSITY0(density0, j, k);
            vol = vol + cell_vol;
            mass = mass + cell_mass;
            ie = ie + cell_mass * ENERGY0(energy0, j, k);
            ke = ke + cell_mass * 0.5 * vsqrd;
            press = press + cell_vol * PRESSURE(pressure, j, k);
        }
    }



    *vl = vol;
    *mss = mass;
    *ien = ie;
    *ken = ke;
    *prss = press;

}
