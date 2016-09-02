
#include <Kokkos_Core.hpp>
#include "../../kernels/viscosity_kernel_c.c"

using namespace Kokkos;

struct viscosity_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt density0,
                pressure,
                viscosity,
                xvel0,
                yvel0;
    field_1d_lt celldx,
                celldy;

    viscosity_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        density0((tile.field.d_density0)),
        pressure((tile.field.d_pressure)),
        viscosity((tile.field.d_viscosity)),

        xvel0((tile.field.d_xvel0)),
        yvel0((tile.field.d_yvel0)),
        celldx((tile.field.d_celldx)),
        celldy((tile.field.d_celldy))
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

            viscosity_kernel_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                celldx,
                celldy,
                density0,
                pressure,
                viscosity,
                xvel0,
                yvel0);
        });
    }
};
