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
 *  @brief C acceleration kernel
 *  @author Wayne Gaudin
 *  @details The pressure and viscosity gradients are used to update the
 *  velocity field.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "definitions_c.h"

void accelerate_kernel_c_(
    struct tile_type *tile,
    double dt)
{
    int x_min = tile->t_xmin;
    int x_max = tile->t_xmax;
    int y_min = tile->t_ymin;
    int y_max = tile->t_ymax;

    double * __restrict__ xarea = tile->field.xarea;
    double * __restrict__ yarea = tile->field.yarea;
    double * __restrict__ volume = tile->field.volume;
    double * __restrict__ density0 = tile->field.density0;
    double * __restrict__ pressure = tile->field.pressure;
    double * __restrict__ viscosity = tile->field.viscosity;
    double * __restrict__ xvel0 = tile->field.xvel0;
    double * __restrict__ yvel0 = tile->field.yvel0;
    double * __restrict__ xvel1 = tile->field.xvel1;
    double * __restrict__ yvel1 = tile->field.yvel1;

    int j, k;

    #pragma omp parallel
    {
        DOUBLEFOR(y_min, y_max + 1, x_min, x_max + 1, ({

            double nodal_mass = (density0[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)] * volume[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]
            + density0[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)] * volume[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)]
            + density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] * volume[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            + density0[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)] * volume[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)])
            * 0.25;
            double stepby_mass_s = 0.5 * dt / nodal_mass;
            xvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
            - stepby_mass_s
            * (xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
            * (pressure[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] - pressure[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)])
            + xarea[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)]
            * (pressure[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)] - pressure[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

            yvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
            - stepby_mass_s
            * (yarea[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            * (pressure[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] - pressure[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)])
            + yarea[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)]
            * (pressure[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)] - pressure[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

            xvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = xvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
            - stepby_mass_s
            * (xarea[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
            * (viscosity[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] - viscosity[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)])
            + xarea[FTNREF2D(j  , k - 1, x_max + 5, x_min - 2, y_min - 2)]
            * (viscosity[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)] - viscosity[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

            yvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)] = yvel1[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
            - stepby_mass_s
            * (yarea[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)]
            * (viscosity[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] - viscosity[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)])
            + yarea[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)]
            * (viscosity[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)] - viscosity[FTNREF2D(j - 1, k - 1, x_max + 4, x_min - 2, y_min - 2)]));

        }));
    }

#ifdef USE_KOKKOS
    Kokkos::fence();
#endif

}
