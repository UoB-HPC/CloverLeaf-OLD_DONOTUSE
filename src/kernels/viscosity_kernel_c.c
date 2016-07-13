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
 *  @brief C viscosity kernel.
 *  @author Wayne Gaudin
 *  @details  Calculates an artificial viscosity using the Wilkin's method to
 *  smooth out shock front and prevent oscillations around discontinuities.
 *  Only cells in compression will have a non-zero value.
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

void viscosity_kernel_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    const double * __restrict__ celldx,
    const double * __restrict__ celldy,
    const double * __restrict__ density0,
    const double * __restrict__ pressure,
    double * __restrict__ viscosity,
    const double * __restrict__ xvel0,
    const double * __restrict__ yvel0)
{
    double ugrad = (xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                    + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                   - (xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                      + xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]);

    double vgrad = (yvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                    + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)])
                   - (yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                      + yvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]);

    double div = (celldx[FTNREF1D(j, x_min - 2)] * (ugrad)
                  + celldy[FTNREF1D(k, y_min - 2)] * (vgrad));

    double strain2 = 0.5 * (xvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]
                            + xvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]
                            - xvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                            - xvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]) / celldy[FTNREF1D(k, y_min - 2)]
                     + 0.5 * (yvel0[FTNREF2D(j + 1, k  , x_max + 5, x_min - 2, y_min - 2)]
                              + yvel0[FTNREF2D(j + 1, k + 1, x_max + 5, x_min - 2, y_min - 2)]
                              - yvel0[FTNREF2D(j  , k  , x_max + 5, x_min - 2, y_min - 2)]
                              - yvel0[FTNREF2D(j  , k + 1, x_max + 5, x_min - 2, y_min - 2)]) / celldx[FTNREF1D(j, x_min - 2)];

    double pgradx = (pressure[FTNREF2D(j + 1, k  , x_max + 4, x_min - 2, y_min - 2)]
                     - pressure[FTNREF2D(j - 1, k  , x_max + 4, x_min - 2, y_min - 2)])
                    / (celldx[FTNREF1D(j, x_min - 2)] + celldx[FTNREF1D(j + 1, x_min - 2)]);
    double pgrady = (pressure[FTNREF2D(j  , k + 1, x_max + 4, x_min - 2, y_min - 2)]
                     - pressure[FTNREF2D(j  , k - 1, x_max + 4, x_min - 2, y_min - 2)])
                    / (celldy[FTNREF1D(k, y_min - 2)] + celldy[FTNREF1D(k + 1, y_min - 2)]);

    double pgradx2 = pgradx * pgradx;
    double pgrady2 = pgrady * pgrady;

    double limiter = ((0.5 * (ugrad) / celldx[FTNREF1D(j, x_min - 2)]) * pgradx2 + (0.5 * (vgrad) / celldy[FTNREF1D(k, y_min - 2)]) * pgrady2 + strain2 * pgradx * pgrady)
                     / MAX(pgradx2 + pgrady2, 1.0e-16);

    if (limiter > 0.0 || div >= 0.0) {
        viscosity[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = 0.0;
    } else {
        pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);
        pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);
        double pgrad = sqrt(pgradx * pgradx + pgrady * pgrady);
        double xgrad = fabs(celldx[FTNREF1D(j, x_min - 2)] * pgrad / pgradx);
        double ygrad = fabs(celldy[FTNREF1D(k, y_min - 2)] * pgrad / pgrady);
        double grad  = MIN(xgrad, ygrad);
        double grad2 = grad * grad;
        viscosity[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] = 2.0 * density0[FTNREF2D(j  , k  , x_max + 4, x_min - 2, y_min - 2)] * grad2 * limiter * limiter;
    }
}
