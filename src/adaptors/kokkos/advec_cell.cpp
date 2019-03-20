
#include <math.h>
#include "../../kernels/ftocmacros.h"
#include "../../kernels/advec_cell_kernel_c.cc"

#include <Kokkos_Core.hpp>

using namespace Kokkos;

struct xsweep_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt pre_vol, post_vol, vol_flux_x,
                vol_flux_y, volume;
    int sweep_number;

    xsweep_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        int _sweep_number
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        pre_vol((tile.field.d_work_array1)), post_vol((tile.field.d_work_array2)),
        vol_flux_x((tile.field.d_vol_flux_x)), vol_flux_y((tile.field.d_vol_flux_y)),
        volume((tile.field.d_volume)),

        sweep_number(_sweep_number)
    {}

    void compute()
    {
        parallel_for("xsweep", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            xsweep(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                pre_vol,
                post_vol,
                volume,
                vol_flux_x,
                vol_flux_y,
                sweep_number);
    }
};


struct ysweep_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt pre_vol, post_vol, vol_flux_x,
                vol_flux_y, volume;
    int sweep_number;

    ysweep_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        int _sweep_number
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        pre_vol((tile.field.d_work_array1)),
        post_vol((tile.field.d_work_array2)),
        vol_flux_x((tile.field.d_vol_flux_x)),
        vol_flux_y((tile.field.d_vol_flux_y)),
        volume((tile.field.d_volume)),

        sweep_number(_sweep_number)
    {}

    void compute()
    {
        parallel_for("ysweep", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            ysweep(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                pre_vol,
                post_vol,
                volume,
                vol_flux_x,
                vol_flux_y,
                sweep_number);
    }
};


struct xcomp1_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt mass_flux_x, ener_flux;
    field_2d_lt vol_flux_x, pre_vol, density1, energy1;
    field_1d_lt vertexdx;

    xcomp1_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        mass_flux_x((tile.field.d_mass_flux_x)),
        ener_flux((tile.field.d_work_array7)),
        vol_flux_x((tile.field.d_vol_flux_x)),
        pre_vol((tile.field.d_work_array1)),
        density1((tile.field.d_density1)),
        energy1((tile.field.d_energy1)),
        vertexdx((tile.field.d_vertexdx))
    {}

    void compute()
    {
        parallel_for("xcomp1", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            xcomp1(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                mass_flux_x,
                ener_flux,
                vol_flux_x,
                pre_vol,
                density1,
                energy1,
                vertexdx);
    }
};



struct ycomp1_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt mass_flux_y, ener_flux;
    field_2d_lt vol_flux_y, pre_vol, density1, energy1;
    field_1d_lt vertexdx;

    ycomp1_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        mass_flux_y((tile.field.d_mass_flux_y)),
        ener_flux((tile.field.d_work_array7)),
        vol_flux_y((tile.field.d_vol_flux_y)),
        pre_vol((tile.field.d_work_array1)),
        density1((tile.field.d_density1)),
        energy1((tile.field.d_energy1)),
        vertexdx((tile.field.d_vertexdx))
    {}

    void compute()
    {
        parallel_for("ycomp1", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {
            ycomp1(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                mass_flux_y,
                ener_flux,
                vol_flux_y,
                pre_vol,
                density1,
                energy1,
                vertexdx);
    }
};


struct xcomp2_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt pre_mass, post_mass, post_ener, advec_vol;
    field_2d_lt density1, energy1;
    field_2d_lt pre_vol, mass_flux_x, ener_flux, vol_flux_x;

    xcomp2_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        pre_mass((tile.field.d_work_array3)),
        post_mass((tile.field.d_work_array4)),
        post_ener((tile.field.d_work_array6)),
        advec_vol((tile.field.d_work_array5)),
        density1((tile.field.d_density1)),
        energy1((tile.field.d_energy1)),
        pre_vol((tile.field.d_work_array1)),
        mass_flux_x((tile.field.d_mass_flux_x)),
        ener_flux((tile.field.d_work_array7)),
        vol_flux_x((tile.field.d_vol_flux_x))
    {}

    void compute()
    {
        parallel_for("xcomp2", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            xcomp2(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                pre_mass,
                post_mass,
                post_ener,
                advec_vol,
                density1,
                energy1,
                pre_vol,
                mass_flux_x,
                ener_flux,
                vol_flux_x);
    }
};


struct ycomp2_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt pre_mass, post_mass, post_ener, advec_vol;
    field_2d_lt density1, energy1;
    field_2d_lt pre_vol, mass_flux_y, ener_flux, vol_flux_y;

    ycomp2_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        pre_mass((tile.field.d_work_array3)),
        post_mass((tile.field.d_work_array4)),
        post_ener((tile.field.d_work_array6)),
        advec_vol((tile.field.d_work_array5)),
        density1((tile.field.d_density1)),
        energy1((tile.field.d_energy1)),
        pre_vol((tile.field.d_work_array1)),
        mass_flux_y((tile.field.d_mass_flux_y)),
        ener_flux((tile.field.d_work_array7)),
        vol_flux_y((tile.field.d_vol_flux_y))
    {}

    void compute()
    {
        parallel_for("ycomp2", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            ycomp2(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                pre_mass,
                post_mass,
                post_ener,
                advec_vol,
                density1,
                energy1,
                pre_vol,
                mass_flux_y,
                ener_flux,
                vol_flux_y);
    }
};
