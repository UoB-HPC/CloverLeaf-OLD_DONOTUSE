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
 *@brief C viscosity kernel.
 *@author Wayne Gaudin
 *@details Calculates an artificial viscosity using the Wilkin's method to
 *  smooth out shock front and prevent oscillations around discontinuities.
 *  Only cells in compression will have a non-zero value.
 */

// #include <stdio.h>
// #include <stdlib.h>
// #include "ftocmacros.h"
// #include <math.h>
// #include "../definitions_c.h"

kernelqual void viscosity_kernel_c_(
    int j, int k,
    int x_min, int x_max, int y_min, int y_max,
    const_field_1d_t celldx,
    const_field_1d_t celldy,
    const_field_2d_t density0,
    const_field_2d_t pressure,
    field_2d_t       viscosity,
    const_field_2d_t xvel0,
    const_field_2d_t yvel0)
{
    double ugrad = (XVEL0(xvel0, j + 1, k)
                    + XVEL0(xvel0, j + 1, k + 1))
                   - (XVEL0(xvel0, j, k)
                      + XVEL0(xvel0, j, k + 1));

    double vgrad = (YVEL0(yvel0, j, k + 1)
                    + YVEL0(yvel0, j + 1, k + 1))
                   - (YVEL0(yvel0, j, k)
                      + YVEL0(yvel0, j + 1, k));

    double div = (FIELD_1D(celldx, j,  x_min - 2) * (ugrad)
                  + FIELD_1D(celldy, k,  y_min - 2) * (vgrad));

    double strain2 = 0.5 * (XVEL0(xvel0, j, k + 1)
                            + XVEL0(xvel0, j + 1, k + 1)
                            - XVEL0(xvel0, j, k)
                            - XVEL0(xvel0, j + 1, k)) / FIELD_1D(celldy, k,  y_min - 2)
                     + 0.5 * (YVEL0(yvel0, j + 1, k)
                              + YVEL0(yvel0, j + 1, k + 1)
                              - YVEL0(yvel0, j, k)
                              - YVEL0(yvel0, j, k + 1)) / FIELD_1D(celldx, j,  x_min - 2);

    double pgradx = (PRESSURE(pressure, j + 1, k)
                     - PRESSURE(pressure, j - 1, k))
                    / (FIELD_1D(celldx, j,  x_min - 2) + FIELD_1D(celldx, j + 1,  x_min - 2));
    double pgrady = (PRESSURE(pressure, j, k + 1)
                     - PRESSURE(pressure, j, k - 1))
                    / (FIELD_1D(celldy, k,  y_min - 2) + FIELD_1D(celldy, k + 1,  y_min - 2));

    double pgradx2 = pgradx * pgradx;
    double pgrady2 = pgrady * pgrady;

    double limiter = ((0.5 * (ugrad) / FIELD_1D(celldx, j,  x_min - 2)) * pgradx2 + (0.5 * (vgrad) / FIELD_1D(celldy, k,  y_min - 2)) * pgrady2 + strain2 * pgradx * pgrady)
                     / MAX(pgradx2 + pgrady2, 1.0e-16);

    if (limiter > 0.0 || div >= 0.0) {
        VISCOSITY(viscosity, j, k) = 0.0;
    } else {
        pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);
        pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);
        double pgrad = sqrt(pgradx * pgradx + pgrady * pgrady);
        double xgrad = fabs(FIELD_1D(celldx, j,  x_min - 2) * pgrad / pgradx);
        double ygrad = fabs(FIELD_1D(celldy, k,  y_min - 2) * pgrad / pgrady);
        double grad = MIN(xgrad, ygrad);
        double grad2 = grad * grad;
        VISCOSITY(viscosity, j, k) = 2.0 * DENSITY0(density0, j, k) * grad2 * limiter * limiter;
    }
}
