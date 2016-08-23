
#include <Kokkos_Core.hpp>
#include "../../kernels/flux_calc_kernel_c.c"

using namespace Kokkos;

struct flux_calc_x_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt xarea,
                xvel0,
                xvel1,
                vol_flux_x;
    double dt;

    flux_calc_x_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        double _dt):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        xarea((tile.field.d_xarea)),
        xvel0((tile.field.d_xvel0)),
        xvel1((tile.field.d_xvel1)),
        vol_flux_x((tile.field.d_vol_flux_x)),
        dt(_dt)
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

            flux_calc_x_kernel(
                j, k,
                x_min, x_max,
                y_min, y_max,
                dt,
                xarea,
                xvel0,
                xvel1,
                vol_flux_x);
        });
    }
};


struct flux_calc_y_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt yarea,
                yvel0,
                yvel1,
                vol_flux_y;
    double dt;

    flux_calc_y_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        double _dt):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        yarea((tile.field.d_yarea)),
        yvel0((tile.field.d_yvel0)),
        yvel1((tile.field.d_yvel1)),
        vol_flux_y((tile.field.d_vol_flux_y)),
        dt(_dt)
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

            flux_calc_y_kernel(
                j, k,
                x_min, x_max,
                y_min, y_max,
                dt,
                yarea,
                yvel0,
                yvel1,
                vol_flux_y);
        });
    }
};

