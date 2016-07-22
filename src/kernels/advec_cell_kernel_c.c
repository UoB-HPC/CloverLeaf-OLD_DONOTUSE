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
 *@brief C cell advection kernel.
 *@author Wayne Gaudin
 *@details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
 */


#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

// #define INRANGE(ymin, ymax, xmin, xmax) ((j >= (xmin)) && (j <= (xmax)) && (k >= (ymin)) && (k <= (ymax)))

void xsweep(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_vol,
    field_2d_t post_vol,
    const_field_2d_t volume,
    const_field_2d_t vol_flux_x,
    const_field_2d_t vol_flux_y,
    int sweep_number)
{
    if (sweep_number == 1) {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + (VOL_FLUX_X(vol_flux_x, j + 1, k)
                                       - VOL_FLUX_X(vol_flux_x, j, k)
                                       + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                       - VOL_FLUX_Y(vol_flux_y, j, k));
        WORK_ARRAY(post_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                     - (VOL_FLUX_X(vol_flux_x, j + 1, k)
                                        - VOL_FLUX_X(vol_flux_x, j, k));
    } else {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                    - VOL_FLUX_X(vol_flux_x, j, k);
        WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    }
}

#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct xsweep_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    View<double**> pre_vol, post_vol, vol_flux_x,
         vol_flux_y, volume;
    int sweep_number;

    xsweep_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        int _sweep_number
    ):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol(*(tile.field.work_array1)), post_vol(*(tile.field.work_array2)),
        vol_flux_x(*(tile.field.vol_flux_x)), vol_flux_y(*(tile.field.vol_flux_y)),
        volume(*(tile.field.volume)),
        sweep_number(_sweep_number)
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            xsweep(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &pre_vol,
                &post_vol,
                &volume,
                &vol_flux_x,
                &vol_flux_y,
                sweep_number);
        });
    }
};

#endif

void ysweep(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_vol,
    field_2d_t post_vol,
    const_field_2d_t volume,
    const_field_2d_t vol_flux_x,
    const_field_2d_t vol_flux_y,
    int sweep_number)
{
    if (sweep_number == 1) {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + (VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                       - VOL_FLUX_Y(vol_flux_y, j, k)
                                       + VOL_FLUX_X(vol_flux_x, j + 1, k)
                                       - VOL_FLUX_X(vol_flux_x, j, k));
        WORK_ARRAY(post_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                     - (VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                        - VOL_FLUX_Y(vol_flux_y, j, k));
    } else {
        WORK_ARRAY(pre_vol, j, k) = VOLUME(volume, j, k)
                                    + VOL_FLUX_Y(vol_flux_y, j, k + 1)
                                    - VOL_FLUX_Y(vol_flux_y, j, k);
        WORK_ARRAY(post_vol, j, k) = VOLUME(volume, j, k);
    }
}

#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct ysweep_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    View<double**> pre_vol, post_vol, vol_flux_x,
         vol_flux_y, volume;
    int sweep_number;

    ysweep_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        int _sweep_number
    ):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol(*(tile.field.work_array1)), post_vol(*(tile.field.work_array2)),
        vol_flux_x(*(tile.field.vol_flux_x)), vol_flux_y(*(tile.field.vol_flux_y)),
        volume(*(tile.field.volume)),
        sweep_number(_sweep_number)
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            ysweep(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &pre_vol,
                &post_vol,
                &volume,
                &vol_flux_x,
                &vol_flux_y,
                sweep_number);
        });
    }
};

#endif

void xcomp1(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mass_flux_x,
    field_2d_t ener_flux,
    const_field_2d_t vol_flux_x,
    const_field_2d_t pre_vol,
    const_field_2d_t density1,
    const_field_2d_t energy1,
    const_field_1d_t vertexdx)
{
    double one_by_six = 1.0 / 6.0;
    int upwind, donor, downwind, dif;
    if (VOL_FLUX_X(vol_flux_x, j, k) > 0.0) {
        upwind = j - 2;
        donor = j - 1;
        downwind = j;
        dif = donor;
    } else {
        upwind = MIN(j + 1, x_max + 2);
        donor = j;
        downwind = j - 1;
        dif = upwind;
    }

    double sigmat = fabs(VOL_FLUX_X(vol_flux_x, j, k) / WORK_ARRAY(pre_vol, donor, k));
    double sigma3 = (1.0 + sigmat) * (FIELD_1D(vertexdx, j,  x_min - 2) / FIELD_1D(vertexdx, dif,  x_min - 2));
    double sigma4 = 2.0 - sigmat;

    double sigma = sigmat;
    double sigmav = sigmat;

    double diffuw = DENSITY1(density1, donor, k) - DENSITY1(density1, upwind, k);
    double diffdw = DENSITY1(density1, downwind, k) - DENSITY1(density1, donor, k);
    double limiter;
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw),
                  one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    MASS_FLUX_X(mass_flux_x, j, k) = VOL_FLUX_X(vol_flux_x, j, k)
                                     * (DENSITY1(density1, donor, k) + limiter);

    double sigmam = fabs(MASS_FLUX_X(mass_flux_x, j, k)) / (DENSITY1(density1, donor, k)
                    * WORK_ARRAY(pre_vol, donor, k));
    diffuw = ENERGY1(energy1, donor, k) - ENERGY1(energy1, upwind, k);
    diffdw = ENERGY1(energy1, downwind, k) - ENERGY1(energy1, donor, k);
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                  , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    WORK_ARRAY(ener_flux, j, k) = MASS_FLUX_X(mass_flux_x, j, k)
                                  * (ENERGY1(energy1, donor, k) + limiter);
}


#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct xcomp1_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    View<double**> mass_flux_x, ener_flux;
    View<double**> vol_flux_x, pre_vol, density1, energy1;
    View<double*> vertexdx;

    xcomp1_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol(*(tile.field.work_array1)), mass_flux_x(*(tile.field.mass_flux_x)),
        ener_flux(*(tile.field.work_array7)), vol_flux_x(*(tile.field.vol_flux_x)),
        density1(*(tile.field.density1)), energy1(*(tile.field.energy1)),
        vertexdx(*(tile.field.vertexdx))
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            xcomp1(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &mass_flux_x,
                &ener_flux,
                &vol_flux_x,
                &pre_vol,
                &density1,
                &energy1,
                &vertexdx);
        });
    }
};

#endif


void ycomp1(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t mass_flux_y,
    field_2d_t ener_flux,
    const_field_2d_t vol_flux_y,
    const_field_2d_t pre_vol,
    const_field_2d_t density1,
    const_field_2d_t energy1,
    const_field_1d_t vertexdy)
{
    double one_by_six = 1.0 / 6.0;
    int upwind, donor, downwind, dif;
    if (VOL_FLUX_Y(vol_flux_y, j, k) > 0.0) {
        upwind = k - 2;
        donor = k - 1;
        downwind = k;
        dif = donor;
    } else {
        upwind = MIN(k + 1, y_max + 2);
        donor = k;
        downwind = k - 1;
        dif = upwind;
    }

    double sigmat = fabs(VOL_FLUX_Y(vol_flux_y, j, k) / WORK_ARRAY(pre_vol, j, donor));
    double sigma3 = (1.0 + sigmat) * (FIELD_1D(vertexdy, k,  y_min - 2) / FIELD_1D(vertexdy, dif,  y_min - 2));
    double sigma4 = 2.0 - sigmat;

    double sigma = sigmat;
    double sigmav = sigmat;

    double diffuw = DENSITY1(density1, j, donor) - DENSITY1(density1, j, upwind);
    double diffdw = DENSITY1(density1, j, downwind) - DENSITY1(density1, j, donor);
    double limiter;
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmav) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                  , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    MASS_FLUX_Y(mass_flux_y, j, k) = VOL_FLUX_Y(vol_flux_y, j, k)
                                     * (DENSITY1(density1, j, donor) + limiter);

    double sigmam = fabs(MASS_FLUX_Y(mass_flux_y, j, k)) / (DENSITY1(density1, j, donor)
                    * WORK_ARRAY(pre_vol, j, donor));
    diffuw = ENERGY1(energy1, j, donor) - ENERGY1(energy1, j, upwind);
    diffdw = ENERGY1(energy1, j, downwind) - ENERGY1(energy1, j, donor);
    if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmam) * SIGN(1.0, diffdw) * MIN(fabs(diffuw), MIN(fabs(diffdw)
                  , one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
    } else {
        limiter = 0.0;
    }
    WORK_ARRAY(ener_flux, j, k) = MASS_FLUX_Y(mass_flux_y, j, k)
                                  * (ENERGY1(energy1, j, donor) + limiter);
}

#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct ycomp1_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    View<double**> mass_flux_y, ener_flux;
    View<double**> vol_flux_y, pre_vol, density1, energy1;
    View<double*> vertexdx;

    ycomp1_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol(*(tile.field.work_array1)), mass_flux_y(*(tile.field.mass_flux_y)),
        ener_flux(*(tile.field.work_array7)), vol_flux_y(*(tile.field.vol_flux_y)),
        density1(*(tile.field.density1)), energy1(*(tile.field.energy1)),
        vertexdx(*(tile.field.vertexdx))
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            ycomp1(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &mass_flux_y,
                &ener_flux,
                &vol_flux_y,
                &pre_vol,
                &density1,
                &energy1,
                &vertexdx);
        });
    }
};

#endif

void xcomp2(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_mass,
    field_2d_t post_mass,
    field_2d_t post_ener,
    field_2d_t advec_vol,
    field_2d_t density1,
    field_2d_t energy1,
    const_field_2d_t pre_vol,
    const_field_2d_t mass_flux_x,
    const_field_2d_t ener_flux,
    const_field_2d_t vol_flux_x)
{
    WORK_ARRAY(pre_mass, j, k) = DENSITY1(density1, j, k)
                                 * WORK_ARRAY(pre_vol, j, k);
    WORK_ARRAY(post_mass, j, k) = WORK_ARRAY(pre_mass, j, k)
                                  + MASS_FLUX_X(mass_flux_x, j, k)
                                  - MASS_FLUX_X(mass_flux_x, j + 1, k);
    WORK_ARRAY(post_ener, j, k) = (ENERGY1(energy1, j, k)
                                   * WORK_ARRAY(pre_mass, j, k)
                                   + WORK_ARRAY(ener_flux, j, k)
                                   - WORK_ARRAY(ener_flux, j + 1, k))
                                  / WORK_ARRAY(post_mass, j, k);
    WORK_ARRAY(advec_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                  + VOL_FLUX_X(vol_flux_x, j, k)
                                  - VOL_FLUX_X(vol_flux_x, j + 1, k);

    DENSITY1(density1, j, k) = WORK_ARRAY(post_mass, j, k) / WORK_ARRAY(advec_vol, j, k);
    ENERGY1(energy1, j, k) = WORK_ARRAY(post_ener, j, k);
}

#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct xcomp2_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    View<double**> pre_mass, post_mass, post_ener, advec_vol;
    View<double**> density1, energy1;
    View<double**> pre_vol, mass_flux_x, ener_flux, vol_flux_x;

    xcomp2_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol(*(tile.field.work_array1)), mass_flux_x(*(tile.field.mass_flux_x)),
        ener_flux(*(tile.field.work_array7)), vol_flux_x(*(tile.field.vol_flux_x)),
        density1(*(tile.field.density1)), energy1(*(tile.field.energy1)),
        pre_mass(*(tile.field.work_array3)), post_mass(*(tile.field.work_array4)),
        post_ener(*(tile.field.work_array6)), advec_vol(*(tile.field.work_array5))
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            xcomp2(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &pre_mass,
                &post_mass,
                &post_ener,
                &advec_vol,
                &density1,
                &energy1,
                &pre_vol,
                &mass_flux_x,
                &ener_flux,
                &vol_flux_x);
        });
    }
};

#endif

void ycomp2(
    int j, int k,
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t pre_mass,
    field_2d_t post_mass,
    field_2d_t post_ener,
    field_2d_t advec_vol,
    field_2d_t density1,
    field_2d_t energy1,
    const_field_2d_t pre_vol,
    const_field_2d_t mass_flux_y,
    const_field_2d_t ener_flux,
    const_field_2d_t vol_flux_y)
{
    WORK_ARRAY(pre_mass, j, k) = DENSITY1(density1, j, k)
                                 * WORK_ARRAY(pre_vol, j, k);
    WORK_ARRAY(post_mass, j, k) = WORK_ARRAY(pre_mass, j, k)
                                  + MASS_FLUX_Y(mass_flux_y, j, k)
                                  - MASS_FLUX_Y(mass_flux_y, j, k + 1);
    WORK_ARRAY(post_ener, j, k) = (ENERGY1(energy1, j, k)
                                   * WORK_ARRAY(pre_mass, j, k)
                                   + WORK_ARRAY(ener_flux, j, k)
                                   - WORK_ARRAY(ener_flux, j, k + 1))
                                  / WORK_ARRAY(post_mass, j, k);
    WORK_ARRAY(advec_vol, j, k) = WORK_ARRAY(pre_vol, j, k)
                                  + VOL_FLUX_Y(vol_flux_y, j, k)
                                  - VOL_FLUX_Y(vol_flux_y, j, k + 1);

    DENSITY1(density1, j, k) = WORK_ARRAY(post_mass, j, k) / WORK_ARRAY(advec_vol, j, k);
    ENERGY1(energy1, j, k) = WORK_ARRAY(post_ener, j, k);
}

#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct ycomp2_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    View<double**> pre_mass, post_mass, post_ener, advec_vol;
    View<double**> density1, energy1;
    View<double**> pre_vol, mass_flux_y, ener_flux, vol_flux_y;

    ycomp2_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol(*(tile.field.work_array1)), mass_flux_y(*(tile.field.mass_flux_y)),
        ener_flux(*(tile.field.work_array7)), vol_flux_y(*(tile.field.vol_flux_y)),
        density1(*(tile.field.density1)), energy1(*(tile.field.energy1)),
        pre_mass(*(tile.field.work_array3)), post_mass(*(tile.field.work_array4)),
        post_ener(*(tile.field.work_array6)), advec_vol(*(tile.field.work_array5))
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            ycomp2(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &pre_mass,
                &post_mass,
                &post_ener,
                &advec_vol,
                &density1,
                &energy1,
                &pre_vol,
                &mass_flux_y,
                &ener_flux,
                &vol_flux_y);
        });
    }
};

#endif


void advec_cell_omp(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number);
void advec_cell_kokkos(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number);

void advec_cell_kernel_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
#ifdef USE_KOKKOS
    advec_cell_kokkos(x_min, x_max, y_min, y_max, tile, dir, sweep_number);
#else
    advec_cell_omp(x_min, x_max, y_min, y_max, tile, dir, sweep_number);
#endif
}

void advec_cell_omp(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
    const_field_1d_t vertexdx = tile.field.vertexdx;
    const_field_1d_t vertexdy = tile.field.vertexdy;
    const_field_2d_t volume = tile.field.volume;
    field_2d_t       density1 = tile.field.density1;
    field_2d_t       energy1 = tile.field.energy1;
    field_2d_t       mass_flux_x = tile.field.mass_flux_x;
    const_field_2d_t vol_flux_x = tile.field.vol_flux_x;
    field_2d_t       mass_flux_y = tile.field.mass_flux_y;
    const_field_2d_t vol_flux_y = tile.field.vol_flux_y;
    field_2d_t       pre_vol = tile.field.work_array1;
    field_2d_t       post_vol = tile.field.work_array2;
    field_2d_t       pre_mass = tile.field.work_array3;
    field_2d_t       post_mass = tile.field.work_array4;
    field_2d_t       advec_vol = tile.field.work_array5;
    field_2d_t       post_ener = tile.field.work_array6;
    field_2d_t       ener_flux = tile.field.work_array7;

    int g_xdir = 1, g_ydir = 2;

    #pragma omp parallel
    {
        if (dir == g_xdir) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                xsweep(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_vol, post_vol, volume, vol_flux_x, vol_flux_y,
                    sweep_number
                );
            });
            DOUBLEFOR(y_min, y_max, x_min, x_max + 2, {
                xcomp1(
                    j,  k,
                    x_min, x_max, y_min, y_max,
                    mass_flux_x, ener_flux, vol_flux_x,
                    pre_vol, density1, energy1, vertexdx
                );
            });

            DOUBLEFOR(y_min, y_max, x_min, x_max, {
                xcomp2(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_mass, post_mass, post_ener, advec_vol,
                    density1, energy1, pre_vol, mass_flux_x,
                    ener_flux, vol_flux_x
                );
            });
        }
        if (dir == g_ydir) {
            DOUBLEFOR(y_min - 2, y_max + 2, x_min - 2, x_max + 2, {
                ysweep(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_vol, post_vol, volume, vol_flux_x, vol_flux_y,
                    sweep_number
                );
            });
            DOUBLEFOR(y_min, y_max + 2, x_min , x_max , {
                ycomp1(
                    j,  k,
                    x_min,  x_max, y_min,  y_max,
                    mass_flux_y, ener_flux, vol_flux_y, pre_vol,
                    density1, energy1, vertexdy
                );
            });
            DOUBLEFOR(y_min, y_max, x_min, x_max, {
                ycomp2(
                    j, k,
                    x_min, x_max, y_min, y_max,
                    pre_mass, post_mass, post_ener, advec_vol, density1,
                    energy1, pre_vol, mass_flux_y, ener_flux, vol_flux_y
                );
            });
        }
    }
}

#ifdef USE_KOKKOS

void advec_cell_kokkos(
    int x_min, int x_max,
    int y_min, int y_max,
    struct tile_type tile,
    int dir,
    int sweep_number)
{
    const_field_1d_t vertexdx = tile.field.vertexdx;
    const_field_1d_t vertexdy = tile.field.vertexdy;
    const_field_2d_t volume = tile.field.volume;
    field_2d_t       density1 = tile.field.density1;
    field_2d_t       energy1 = tile.field.energy1;
    field_2d_t       mass_flux_x = tile.field.mass_flux_x;
    const_field_2d_t vol_flux_x = tile.field.vol_flux_x;
    field_2d_t       mass_flux_y = tile.field.mass_flux_y;
    const_field_2d_t vol_flux_y = tile.field.vol_flux_y;
    field_2d_t       pre_vol = tile.field.work_array1;
    field_2d_t       post_vol = tile.field.work_array2;
    field_2d_t       pre_mass = tile.field.work_array3;
    field_2d_t       post_mass = tile.field.work_array4;
    field_2d_t       advec_vol = tile.field.work_array5;
    field_2d_t       post_ener = tile.field.work_array6;
    field_2d_t       ener_flux = tile.field.work_array7;
    int g_xdir = 1, g_ydir = 2;

    if (dir == g_xdir) {
        xsweep_functor f1(
            tile,
            x_min - 2, x_max + 2, y_min - 2, y_max + 2,
            sweep_number);
        f1.compute();

        xcomp1_functor f2(
            tile,
            x_min, x_max + 2, y_min, y_max);
        f2.compute();

        xcomp2_functor f3(
            tile,
            x_min, x_max, y_min, y_max);
        f3.compute();
        Kokkos::fence();
    }
    if (dir == g_ydir) {
        ysweep_functor f1(
            tile,
            x_min - 2, x_max + 2, y_min - 2, y_max + 2,
            sweep_number);
        f1.compute();

        ycomp1_functor f2(
            tile,
            x_min, x_max, y_min, y_max + 2);
        f2.compute();

        ycomp2_functor f3(
            tile,
            x_min, x_max, y_min, y_max);
        f3.compute();
        Kokkos::fence();
    }
}
#endif